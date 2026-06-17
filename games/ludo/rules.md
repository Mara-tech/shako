# Ludo (Les Petits Chevaux)

## Players

2 to 4 players. Each player controls 4 horses of their colour.

## Object of the game

Be the first player to move all 4 of your horses from the stable to the home square.

## Board

The board has a circular main track of `track_size` squares (default 52, but configurable as any positive multiple of 4). The four entry squares are evenly spaced: at positions 0, track_size/4, track_size/2, and 3·track_size/4. Each player also has a private home column of 6 squares at the end of their path.

## How to play

At the start of each turn, a die is rolled automatically. The current player then chooses which horse to move.

**Exiting the stable:** A horse leaves the stable only on a roll of 6. It is placed on the player's entry square (position 0 on their path).

**Moving on the main track:** On any roll, a horse on the main track advances exactly the rolled number of squares along its path.

**Entering the home column:** A horse can only enter the home column from the last main-track square (position track_size − 1 on its path). The home column squares are numbered 1 to 6; rolling i from that last square places the horse on home-column square i directly. A horse that would overshoot position track_size − 1 into the home column cannot make that move — it must first reach that square exactly.

**No overshoot:** A horse cannot advance beyond the home square (position track_size + 5). If the die roll would carry it past, the move is illegal.

**Rolling a 6:** After successfully moving a horse on a roll of 6, the same player rolls again and takes a bonus turn. If no legal move is available (forced pass), the turn passes to the next player without a bonus roll.

**Capturing:** Landing on a square occupied by an opponent's horse sends that horse back to its stable. Exception: a horse standing on its own entry square cannot be captured. Horses in the home column are also safe.

## End of the game

The first player to move all 4 horses to the home square (position track_size + 5 on their path) wins immediately. Final scores: winner = 1.0, all others = 0.0. Partial (mid-game) scores reflect the fraction of horses that have reached home.

## Notes on this implementation

- Three consecutive rolls of 6 do **not** forfeit the turn (simplified rule).
- Safe squares are entry squares only; there are no additional rosette/shelter squares.
- 2- and 3-player games use the same board with only the first `n_players` entry points active.
