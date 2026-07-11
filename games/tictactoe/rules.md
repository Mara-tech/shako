# Tic Tac Toe

## Players
2 players (0 and 1) who alternate turns. Player 0 goes first.

## OBJECT OF THE GAME

Be the first player to place three of your marks in a horizontal, vertical, or diagonal row on the 3×3 grid.

## HOW TO PLAY

The game is played on a 3×3 grid. One player is assigned **X** and the other **O**. Players take turns placing their mark on any empty cell, starting with player 0.

### END OF THE GAME

The game ends immediately when one player occupies three cells in a row — horizontally, vertically, or along either diagonal. That player wins (score 1.0); the other player scores 0.0. If all nine cells are filled without a winner, the game is a draw and both players score 0.5.

## Edge cases

- Each call to `get_initial_state` starts a fresh, independent game — this adapter has no notion of a multi-round match. Playing a best-of-N series against the same opponent (with a running scoreboard) is a concern of the caller — e.g. the CLI's "play again" loop — not of the adapter, so that the seat/starter advantage of any individual game stays visible to analysis tools instead of being averaged away inside a single session.
- Since player 0 always starts, giving each side a fair shot at the first move is a matter of which player occupies seat 0 — the caller's job (e.g. the CLI's seat prompt, or `SelfPlayTrainer`'s seat alternation between candidate and current agent), not the adapter's.
