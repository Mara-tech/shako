# AOT Reconquête

A turn-based strategy game set in the Attack on Titan universe. The rules are not implemented
in this repository: they run in TypeScript, in the `aot-reconquete.js` repository, and this
game's adapter relays shako's questions to them over a JSON stdio bridge. What follows
summarises the rules the adapter plays against; the authority is `rules_and_design.md` in
that repository.

## Players

**One.** The player commands the Scout forces. The titans are part of the game rather than an
opponent that chooses: they advance on their own between the player's turns, by rules the
game applies. So `get_n_players()` is 1 and `get_current_player()` is always seat 0.

Nothing on the board is hidden from that player, so the observable state is the whole state.

## The board

A 9x12 grid of tiles. Tiles carry a terrain type: walls, towns, villages, forest, field, and
**gates** — which are either functional (`door`) or breached (`broken_door`). Scout forces and
titan swarms each occupy a tile; a tile holds at most one force and one swarm.

## A turn

Every Scout force may act, then the titans advance towards their objective — the nearest unit
— and the turn counter goes up. A fight starts when scouts and titans end up on the same tile.

Each force has **action points**, and nearly everything costs one: moving, engaging a fight
(engaging is a move, and is paid for as one), and each side action. A force at zero action
points can only wait for the turn to end. Moving also consumes gas where the terrain demands
the ODM gear.

## Actions

The adapter takes the game's own vocabulary as it is; each action is plain JSON carrying its
own parameters, and `Action.data` holds it unchanged.

**Board actions** name the tile the acting force stands on and the adjacent tile acted upon:

- `move` — step onto a free adjacent tile.
- `join_force` — step onto an adjacent allied force, merging into it.
- `attack` — engage the titan swarm on an adjacent tile, which opens a combat.

**Side actions** name the tile the force acts from, and four of them also name one member,
equipment being carried per scout: `heal_rest`, `change_gas`, `change_blades`, `repair_door`,
`build_outpost`, `find_horse`, `leave_horse`.

**Combat actions**, offered only while a fight runs, and then they are the *only* actions
offered: `combat_attack` (naming the body part struck, which decides how many dice are
thrown), `leave_combat`, `capture_titan`.

**`end_of_turn`** ends the Scouts' turn and lets the titans play. It costs nothing.

### The list is never empty on a live position

`BaseAdapter` requires a non-empty list of legal actions on any non-terminal state, and this
game honours it in two halves: out of combat, `end_of_turn` is always legal — so a player
whose forces have all spent their points is never stuck — and in combat, the capture is always
offered. Verified against the running game: from the opening position, playing every action
but the end of turn until the forces are exhausted leaves exactly
`[{"kind": "end_of_turn"}]`, never an empty list.

## End of the game

- **Victory** — at least **4 gates are functional**. A gate is functional or breached; the
  model keeps no third state, so "repaired" is read as "functional".
- **Defeat** — a titan holds a **breached gate on the uppermost gate line** with **no scout on
  that tile**. A scout sharing the tile is making a last stand as the gate falls, so the city
  is not lost yet.
- Defeat is checked first: a titan holding the last gate is a breach that gates repaired
  elsewhere do not buy back.

**There is no draw.** The game's outcome has three values — ongoing, victory, defeat — and no
fourth, so shako's rule that a draw must score strictly between a win and a loss has nothing
to bite on here.

## Scores

One player, so one score, and it is the game's own scale:

- **defeat — 0.0**
- **victory — 1 / turnCounter**, so a game won in ten turns scores ten times one won in a
  hundred, and a search is pushed to win quickly. The counter is 1 on the first turn, so a
  victory scores in `]0, 1]`.
- **still ongoing — 0.0**. Intermediate positions are not estimated.

## Running it

The adapter needs a checkout of the game with its dependencies installed:

```bash
git clone <the aot-reconquete.js repository>
cd aot-reconquete.js && npm ci
```

It is looked for in the adapter's `game_dir` argument, then in `$AOT_RECONQUETE_DIR`, then as
a sibling directory of the shako checkout. The adapter launches and stops the Node process
itself; nothing has to be started by hand.
