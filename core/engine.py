from __future__ import annotations

import multiprocessing
import random
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError

from core.base_adapter import BaseAdapter
from core.base_agent import BaseAgent
from core.types import Action, GameResult


_TIMEOUT_SENTINEL = "__shako_timeout__"


def _run_one_game(engine: SimulationEngine) -> GameResult:
    """Top-level worker for `multiprocessing.Pool` (must be picklable)."""
    return engine.run_game()


class SimulationEngine:
    """Drives a single adapter+agents combination through one or many games.

    The engine owns the turn-by-turn loop:
      1. ask the adapter who plays next,
      2. fetch that player's observable state and legal actions,
      3. ask the agent to pick an action (optionally under a timeout),
      4. validate it against the legal set,
      5. apply it via the adapter, repeat until terminal.

    Defensive behaviour:
      * Illegal actions (including timeouts) are silently replaced by a uniform
        random legal action, and counted per-player in the resulting `GameResult`.
      * Games that exceed `max_turns` are terminated with all scores set to 0.0
        and `timed_out=True`.
    """

    def __init__(
        self,
        adapter: BaseAdapter,
        agents: list[BaseAgent],
        max_action_ms: float | None = None,
        max_turns: int = 10_000,
        record: bool = False,
        seed: int | None = None,
    ) -> None:
        """Args:
            adapter: the game implementation.
            agents: one agent per seat; `len(agents)` must equal `adapter.get_n_players()`.
            max_action_ms: per-action wall-clock budget in milliseconds. `None`
                disables the timeout (no executor is spawned).
            max_turns: hard cap on the number of turns per game.
            record: when True, every applied action is appended to
                `GameResult.actions` as `(turn_index, player_id, action)`.
            seed: seed for the engine's tiebreak RNG (used when replacing
                illegal actions). The adapter and agents manage their own RNGs.
        """
        if len(agents) != adapter.get_n_players():
            raise ValueError(
                f"adapter expects {adapter.get_n_players()} agents, got {len(agents)}"
            )
        self.adapter = adapter
        self.agents = agents
        self.max_action_ms = max_action_ms
        self.max_turns = max_turns
        self.record = record
        self._rng = random.Random(seed)

    def run_game(self) -> GameResult:
        """Play one full game and return its `GameResult`."""
        n_players = self.adapter.get_n_players()
        for pid, agent in enumerate(self.agents):
            agent.on_game_start(pid, n_players)

        state = self.adapter.get_initial_state()
        recording: list[tuple[int, int, Action]] = []
        illegal_counts: dict[int, int] = {pid: 0 for pid in range(n_players)}
        timed_out = False

        executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=1) if self.max_action_ms is not None else None
        )

        start = time.perf_counter()
        n_turns = 0
        try:
            while not self.adapter.is_terminal(state):
                if n_turns >= self.max_turns:
                    timed_out = True
                    break

                pid = self.adapter.get_current_player(state)
                legal = self.adapter.get_legal_actions(state, pid)
                if not legal:
                    # Defensive: no legal action in a non-terminal state.
                    timed_out = True
                    break

                obs = self.adapter.get_observable_state(state, pid)
                action = self._ask_agent(self.agents[pid], obs, legal, executor)

                if action not in legal:
                    illegal_counts[pid] += 1
                    action = self._rng.choice(legal)

                if self.record:
                    recording.append((n_turns, pid, action))

                state = self.adapter.apply_action(state, action, pid)
                n_turns += 1
        finally:
            if executor is not None:
                executor.shutdown(wait=False)

        duration_ms = (time.perf_counter() - start) * 1000.0

        if timed_out:
            scores: dict[int, float] = {pid: 0.0 for pid in range(n_players)}
            winner_id: int | None = None
        else:
            scores = self.adapter.get_scores(state)
            winner_id = self._pick_winner(scores)

        for agent in self.agents:
            agent.on_game_end(scores)

        return GameResult(
            scores=scores,
            n_turns=n_turns,
            winner_id=winner_id,
            duration_ms=duration_ms,
            illegal_action_counts=illegal_counts,
            actions=recording if self.record else None,
            timed_out=timed_out,
        )

    def run_batch(self, n_games: int, n_workers: int = 1) -> list[GameResult]:
        """Play `n_games` games and return all results.

        With `n_workers > 1`, games run in parallel processes via
        `multiprocessing` (not threads — we want to dodge the GIL for CPU-bound
        RL/MCTS agents). The engine, adapter, and agents must therefore be
        picklable. Each worker receives a copy of the engine, so any learning
        state updated inside `on_game_end` will NOT propagate back to the
        parent — gather results and update outside the batch if needed.
        """
        if n_games <= 0:
            return []
        if n_workers <= 1:
            return [self.run_game() for _ in range(n_games)]

        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            return pool.map(_run_one_game, [self] * n_games)

    def _ask_agent(
        self,
        agent: BaseAgent,
        obs: object,
        legal: list[Action],
        executor: ThreadPoolExecutor | None,
    ) -> Action:
        """Invoke `agent.choose_action`, enforcing `max_action_ms` if set.

        On timeout, returns a sentinel Action that will be rejected by the
        caller's legality check and replaced by a random legal action. Note
        that Python cannot truly kill the agent's worker thread — it keeps
        running in the background until it finishes on its own. This is a
        deliberate trade-off for cross-platform portability.
        """
        if executor is None or self.max_action_ms is None:
            return agent.choose_action(obs, legal)  # type: ignore[arg-type]
        future = executor.submit(agent.choose_action, obs, legal)  # type: ignore[arg-type]
        try:
            return future.result(timeout=self.max_action_ms / 1000.0)
        except FutureTimeoutError:
            return Action(data={_TIMEOUT_SENTINEL: True})

    @staticmethod
    def _pick_winner(scores: dict[int, float]) -> int | None:
        """Return the unique highest-scoring player, or None on a tie."""
        if not scores:
            return None
        top = max(scores.values())
        winners = [pid for pid, s in scores.items() if s == top]
        return winners[0] if len(winners) == 1 else None
