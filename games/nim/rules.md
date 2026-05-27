# Nim

## Players
2 players

## OBJECT OF THE GAME

Force your opponent to take the last stick (misère variant, default), or be the player who takes the last stick (standard variant).

## HOW TO PLAY

The game begins with a single pile of sticks (21 by default). Players alternate turns. On each turn the current player removes between 1 and `max_take` sticks (default 3) from the pile. The number of sticks taken cannot exceed the remaining pile size.

### END OF THE GAME

The game ends when the pile is empty.

- **Misère variant** (`last_takes_wins = False`, default): the player who takes the final stick **loses**.
- **Standard variant** (`last_takes_wins = True`): the player who takes the final stick **wins**.
