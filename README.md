# chess-blunder-predictor
Predicts the probability that a chess player blunders on their next move, using the position, ratings, side to move, and clock time.

# Key results
- The best model was a tuned LightGBM. Reached **ROC-AUC of 0.726** and **PR-AUC of 0.172** on 3,000+ unseen games. About 8% of moves are blunders, which means that random guessing scores a PR-AUC of about 0.08. The model is roughly twice as good as chance at ranking risky moves.
- **Material difference was the most useful feature:** This feature lifted the AUC of a model without the board from 0.556 to 0.661.
- **A tuned neural network couldn't beat tuned LightGBM,** tuned neural network scored 0.712 ROC-AUC and tuned LightGBM scored 0.726 ROC-AUC. Both of the models were completely different type of models but they reached a similar level. This suggests the limit is the information in the inputs, not the models. Though there is a potential for a CNN to score much higher. The neural network I used was an MLP.

# Data
- **Source:** [Lichess open database](https://database.lichess.org/), January 2024, first 1 GB of the compressed file.
- **Filters:** rated blitz game, standard chess, both players rated between 1000 and 2000, engine evaluations and clock times present.
- **Size:** 20,000 games, resulting in 1,090,155 moves total. Skipped the first 5 moves on each game because its mostly opening theory.
- **Blunder rate:** 8.1% of all moves.

# How a blunder is defined
A blunder is a move that results in a 15 or more percentage drop of win probability for the mover. This is the blunder threshold that Lichess uses. Win probabily is computed using [lichess's published formula](https://lichess.org/page/accuracy).

# Features (total of 778 features)
Every feature is known **before** the move is played. No engine evaluation was used as an input which means that the model could run on a live position with no engine.
- **Board:** 12 piece types x 64 squares, one-hot encoded (768 values).
- **Side to move:** 1 for White, 0 for black.
- **Clock scaled:** (Moving side's time left before the move) / (base time + 40 * increment)
- **Base time scaled:** (base time) / 300
- **Time increment scaled:** (increment) / 5
- **Mover and opponent rating scaled:** (rating - 1500) / 500
- **Material difference scaled:** max(-1.0, min(1.0, (material_diff / 39)))
- **Number of legal moves scaled:** max(-1.0, min(1.0, (n_legal / 80))),
- **Number hanging pieces for both sides scaled:** min(1, hanging_count / 15)

# Method:
- **Split:** train / validation / test = 70 / 15 / 15. Split by game using GroupShuffleSplit which means that moves from one game won't appear in different sides. This ensures that the model doesn't recognize games it has seen.
- **Metrics:** ROC-AUC and PR-AUC. Didn't use Accuracy because always guessing "no blunder" would be right 92% of the time.
- **Models:** LightGBM (default), LightGBM tuned with Optuna (30 trials and early stopping on the validation set), and an MLP (778 inputs and three hidden layers) tuned with 30 trials of Optuna.
- The test set was scored once per model after the tuning was finished.

# Results:
| **Model** | **ROC-AUC** | **PR-AUC** |
| --------- | ----------- | ---------- |
| Random Guessing | 0.500  | 0.081      |
| LightGBM (default) | 0.7095 | 0.1613    |
| MLP (tuned) | 0.7120 | 0.1647 |
| **LightGBM (tuned)** | **0.7256** | **0.1720** |

# What each feature group adds to result (LightGBM default settings)
| **Features used** | **ROC-AUC** | **PR-AUC** |
| ----------------- | ----------- | ---------- |
| Side to move, clock features, ratings, material difference, number of legal moves, and number of hanging pieces | 0.689 | 0.148 |
| Board only | 0.623 | 0.120 |
| Everything | 0.712 | 0.163 |
