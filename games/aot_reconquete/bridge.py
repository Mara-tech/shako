"""Stdio client for the AOT Reconquête Node bridge.

The game's rules run in TypeScript, in the `aot-reconquete.js` repository. That repository
ships a Node process that holds game states and answers JSON requests, one object per line
each way — its protocol is documented there in `shako-bridge.md`. This module is the Python
half of that pipe: launching the process, one request/one response, and stopping it cleanly.

It knows the *transport* and nothing about shako: mapping the protocol onto `BaseAdapter`
is `adapter.py`'s job. That split mirrors the Node side (`ShakoBridge.ts` transports,
`ShakoAdapter.ts` names the methods) and is the arbitration the game's repository already
made: the Node side stays close to its own domain, the adaptation to shako's signatures is
done here, where it costs less to correct when that base class moves.

Three behaviours worth stating, because each one was a decision:

* **The process is launched directly, not through `npm run shako-bridge`.** `npm` prints a
  two-line banner on *stdout* before the script starts, and stdout carries the protocol —
  a client following the documented command reads `> aot-reconquete@0.0.1 shako-bridge` as
  its first response. `npm run --silent` is clean, but a bare `node` invocation of ts-node's
  entry point depends on neither a shell nor PATH, which is also what the game's own
  reference client (`spawnShakoBridge` in `ShakoBridgeClient.ts`) does.
* **stderr is drained by a thread.** The game talks — `Starting turn #n` on every turn, a
  `WARNING : is this expected ?` at each controller guard — and the bridge sends all of it to
  stderr so that stdout stays the protocol's. Nobody reading that pipe means it fills, and a
  full pipe blocks the Node process mid-answer. The tail is kept so that a crash can be
  quoted in the exception instead of being lost.
* **Closing stdin stops the bridge**, which is the protocol's own shutdown: there is no
  `quit` method. That also means an orphaned process ends by itself if this interpreter
  dies without unwinding — the pipe closes with it.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import weakref
from collections import deque
from pathlib import Path
from typing import Any, Self

# Where the game repository is looked for when the caller names none.
GAME_DIR_ENV_VAR = "AOT_RECONQUETE_DIR"
_SIBLING_DIR_NAME = "aot-reconquete.js"

# The three paths that make up the documented launch, relative to the game repository.
_TS_NODE_BIN = Path("node_modules") / "ts-node" / "dist" / "bin.js"
_TS_CONFIG = Path("tsconfig.headless.json")
_BRIDGE_ENTRY_POINT = Path("src") / "game" / "headless" / "shakoBridgeMain.ts"

# The wire format this client was written against — `describe` answers it, and a mismatch is
# refused at the handshake rather than misread request by request.
SUPPORTED_PROTOCOL = 1

# How much of the game's chatter is kept to quote in an exception. A crash's stack is the
# last thing written, so keeping the tail keeps the useful half.
_STDERR_TAIL_LINES = 60

# How long a stopping process is given before it is signalled. It exits on EOF as soon as it
# finishes the request in flight, so this only matters when one is stuck.
_SHUTDOWN_TIMEOUT_S = 10.0


class AotReconqueteError(RuntimeError):
    """Anything this game's Python side refuses to carry on from."""


class BridgeProcessError(AotReconqueteError):
    """The Node process could not be launched, or died while it was being talked to."""


class BridgeCallError(AotReconqueteError):
    """The bridge answered `ok: false` — a request it knows and would not serve.

    Carries the protocol's own error code (`invalid_params`, `unknown_state`,
    `internal_error`…) so a caller can branch on it instead of on a message.
    """

    def __init__(self, method: str, code: str, message: str) -> None:
        super().__init__(f"{method} failed [{code}]: {message}")
        self.method = method
        self.code = code
        self.bridge_message = message


class BridgeProtocolError(AotReconqueteError):
    """The bridge answered something the protocol does not describe.

    A line that is not JSON, a response whose `id` matches no request, or an answer that
    breaks an invariant the protocol states. It is deliberately distinct from
    `BridgeCallError`: this one says the *bridge* is wrong, not the request.
    """


def resolve_game_dir(game_dir: str | os.PathLike[str] | None = None) -> Path:
    """Locate the `aot-reconquete.js` checkout, and check it can be launched.

    In order: the argument, then `$AOT_RECONQUETE_DIR`, then a sibling of the shako
    repository. Raises `BridgeProcessError` naming what is missing and how to fix it —
    a forgotten `npm ci` is the common case and is worth a message rather than a traceback
    out of `subprocess`.
    """
    if game_dir is not None:
        root = Path(game_dir).expanduser()
        origin = "the game_dir argument"
    elif os.environ.get(GAME_DIR_ENV_VAR):
        root = Path(os.environ[GAME_DIR_ENV_VAR]).expanduser()
        origin = f"${GAME_DIR_ENV_VAR}"
    else:
        root = Path(__file__).resolve().parents[2].parent / _SIBLING_DIR_NAME
        origin = "the default (a sibling of the shako checkout)"

    root = root.resolve()
    if not (root / _BRIDGE_ENTRY_POINT).is_file():
        raise BridgeProcessError(
            f"no AOT Reconquête checkout at {root} (from {origin}): "
            f"{_BRIDGE_ENTRY_POINT} is missing. Clone the game repository and point "
            f"${GAME_DIR_ENV_VAR} at it, or pass game_dir=..."
        )
    if not (root / _TS_NODE_BIN).is_file():
        raise BridgeProcessError(
            f"{root} has no {_TS_NODE_BIN}: the game's dependencies are not installed. "
            f"Run `npm ci` there first."
        )
    return root


def is_game_available(game_dir: str | os.PathLike[str] | None = None) -> bool:
    """True when `resolve_game_dir` would succeed — for tests that skip without the game."""
    try:
        resolve_game_dir(game_dir)
    except BridgeProcessError:
        return False
    return True


class ShakoBridge:
    """One Node process, and the one-request/one-response exchange over its pipes.

    Started on the first call rather than in `__init__`, so an instance costs nothing until
    it is used — which is what lets the adapter hand one out to a multiprocessing worker that
    may never need it.

    **Every exchange is taken under a lock.** The engine runs an agent under a
    `ThreadPoolExecutor` when `max_action_ms` is set, and an agent that timed out keeps
    running: its next call would interleave a request with the main thread's on the same
    pipe, and both would read the other's answer. The lock makes each write/read atomic; the
    bridge itself serialises its handlers, so nothing else is needed.
    """

    def __init__(self, game_dir: str | os.PathLike[str] | None = None, node: str = "node") -> None:
        """Args:
            game_dir: the `aot-reconquete.js` checkout. `None` resolves it as documented in
                `resolve_game_dir`.
            node: the Node executable. Overridable for an environment where it is not on PATH.
        """
        self.game_dir = resolve_game_dir(game_dir)
        self.node = node
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._finalizer: weakref.finalize | None = None
        self._stderr_tail: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)
        self._next_id = 0
        # Read from `describe` at the handshake; the adapter asserts them against what it
        # hard-codes, so a one-player game growing a second seat is noticed here.
        self.players = 0
        self.current_player = 0

    # -------------------------------------------------------------- life cycle

    def start(self) -> None:
        """Launch the process and shake hands. Idempotent."""
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return
            self._process = self._spawn()
            # Registered on the process and not on `self`: a finalizer holding `self` would
            # keep it alive for ever. `weakref.finalize` also runs at interpreter exit, so
            # this covers a garbage-collected bridge and a program that never calls close().
            self._finalizer = weakref.finalize(self, _stop_process, self._process)
            try:
                self._handshake()
            except Exception:
                self.close()
                raise

    def close(self) -> None:
        """Close the pipe and wait for the process to go. Idempotent, and safe at exit."""
        with self._lock:
            if self._finalizer is not None:
                self._finalizer()  # runs `_stop_process` exactly once
                self._finalizer = None
            self._process = None

    def is_running(self) -> bool:
        """True while the Node process is alive."""
        return self._process is not None and self._process.poll() is None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------- calls

    def call(self, method: str, **params: Any) -> dict[str, Any]:
        """Send one request, return its `result`.

        Raises `BridgeCallError` on an `ok: false` answer, `BridgeProcessError` when the
        process is gone, and `BridgeProtocolError` on anything else off the wire.
        """
        with self._lock:
            self.start()
            response = self._exchange(method, params)
            if response.get("ok") is True:
                result = response.get("result")
                if not isinstance(result, dict):
                    raise BridgeProtocolError(f"{method}: result is not an object: {result!r}")
                return result
            error = response.get("error")
            if not isinstance(error, dict):
                raise BridgeProtocolError(
                    f"{method}: a failure carries an error object: {response!r}"
                )
            raise BridgeCallError(method, str(error.get("code")), str(error.get("message")))

    def stderr_tail(self) -> str:
        """The last lines the process wrote to stderr — its chatter, and any crash."""
        return "\n".join(self._stderr_tail)

    # ---------------------------------------------------------------- internals

    def _spawn(self) -> subprocess.Popen[str]:
        argv = [
            self.node,
            str(self.game_dir / _TS_NODE_BIN),
            "--project",
            str(self.game_dir / _TS_CONFIG),
            str(self.game_dir / _BRIDGE_ENTRY_POINT),
        ]
        try:
            process = subprocess.Popen(
                argv,
                cwd=str(self.game_dir),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,  # line-buffered: a request must reach the bridge when it is written
            )
        except OSError as e:
            raise BridgeProcessError(
                f"cannot launch the AOT Reconquête bridge ({argv[0]}): {e}"
            ) from e
        self._stderr_tail.clear()
        threading.Thread(
            target=_drain, args=(process.stderr, self._stderr_tail), daemon=True
        ).start()
        return process

    def _handshake(self) -> None:
        """Read the constants once, and refuse a wire format this client cannot speak."""
        described = self.call("describe")
        protocol = described.get("protocol")
        if protocol != SUPPORTED_PROTOCOL:
            raise BridgeProtocolError(
                f"the bridge speaks protocol {protocol}, this client speaks {SUPPORTED_PROTOCOL}"
            )
        self.players = int(described.get("players", 0))
        self.current_player = int(described.get("currentPlayer", 0))

    def _exchange(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        process = self._process
        if process is None or process.poll() is not None:
            raise BridgeProcessError(f"the bridge is not running ({self._death_note()})")
        self._next_id += 1
        request_id = self._next_id
        payload = json.dumps({"id": request_id, "method": method, "params": params})
        try:
            process.stdin.write(payload + "\n")  # type: ignore[union-attr]
            process.stdin.flush()  # type: ignore[union-attr]
        except (BrokenPipeError, OSError) as e:
            raise BridgeProcessError(
                f"{method}: the bridge closed its input ({self._death_note()})"
            ) from e

        line = process.stdout.readline()  # type: ignore[union-attr]
        if line == "":
            raise BridgeProcessError(
                f"{method}: the bridge answered nothing ({self._death_note()})"
            )
        try:
            response = json.loads(line)
        except json.JSONDecodeError as e:
            raise BridgeProtocolError(
                f"{method}: the bridge wrote a line that is not JSON: {line!r}"
            ) from e
        if not isinstance(response, dict):
            raise BridgeProtocolError(f"{method}: a response is a JSON object, got {response!r}")
        if response.get("id") != request_id:
            raise BridgeProtocolError(
                f"{method}: answer to request {response.get('id')!r} "
                f"while {request_id} was asked — "
                f"the stream is out of step"
            )
        return response

    def _death_note(self) -> str:
        """Why the process is gone, as far as it said so — quoted into every failure."""
        code = self._process.poll() if self._process is not None else None
        tail = self.stderr_tail()
        state = "exit code " + str(code) if code is not None else "still running"
        return f"{state}; stderr tail:\n{tail}" if tail else state


def _drain(stream: Any, sink: deque[str]) -> None:
    """Keep stderr moving, and keep its tail. A pipe nobody reads blocks the writer."""
    try:
        for line in stream:
            sink.append(line.rstrip("\n"))
    except (ValueError, OSError):
        return  # the stream was closed under us, which is how a stopped bridge ends this


def _stop_process(process: subprocess.Popen[str]) -> None:
    """Stop the bridge the way its protocol says: close its stdin, then wait.

    A module-level function, taking the process alone, so that `weakref.finalize` can hold it
    without holding the `ShakoBridge` that made it. Escalates only if the process does not go:
    the documented shutdown is the EOF, a signal is the fallback.
    """
    if process.poll() is None:
        try:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
        except OSError:
            pass
        try:
            process.wait(timeout=_SHUTDOWN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=_SHUTDOWN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    for pipe in (process.stdin, process.stdout, process.stderr):
        try:
            if pipe is not None and not pipe.closed:
                pipe.close()
        except OSError:
            pass
