# Connect Four

## Board

A 6-row × 7-column grid, initially empty.

## Players

2 players (0 and 1) who alternate turns. Player 0 goes first.

## Actions

On each turn the active player chooses a non-full column (0–6).
A disc drops into that column and occupies the lowest empty cell.

## End of game

The game ends as soon as one of the following conditions is met:

- **Win**: a player connects four discs in a row — horizontally, vertically,
  or diagonally. That player wins (score 1.0); the other loses (score 0.0).
- **Draw**: the board is full with no four-in-a-row.
  Both players receive a score of 0.5.

## Edge cases

- A column whose top cell (row 0) is occupied is full and does not appear
  in the list of legal actions.
- `get_legal_actions` only returns an empty list on a terminal state
  (the game ends before all columns are full when a player wins first).
