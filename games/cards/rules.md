# Cards

## Players
2 players

## OBJECT OF THE GAME

Win the most tricks by the time both hands are empty.

## HOW TO PLAY

A 20-card deck is used (values 1–10, two copies of each). Each player is dealt 5 cards (configurable via `hand_size`), kept secret from the opponent. The remaining 10 cards are set aside and unknown to both players.

Play proceeds in tricks of 2 cards. The trick leader plays a card face-up; the opponent then plays a card knowing what was led.

- The player with the **higher card** wins the trick, scores **+1 point**, and leads the next trick.
- On a **tie**, both cards are discarded and no point is awarded; the previous leader leads again.

Each player's hand is hidden from the opponent. The set-aside cards are unknown to both — any agent reasoning about unseen cards must sample over this unknown pool.

### END OF THE GAME

The game ends when both players have played all their cards. The player with the higher score wins. If scores are equal, the game is a draw.
