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

I expected the neural network to win given how chess has complex patterns. I am guessing that 765k positions were just not enough for the neural network to learn the one-hot encoded board.

# What each feature group adds to result (LightGBM default settings)
Features were added one at a time on top of the OHE board. These runs used default LightGBM on a 75/25 split by game.
| **Features used** | **ROC-AUC** | **PR-AUC** |
| ----------------- | ----------- | ---------- |
| Board only | 0.6230 | 0.1192 |
| + side to move | 0.6227 | 0.1190 |
| + mover's clock | 0.6223 | 0.1193 |
| + base time | 0.6232 | 0.1202 |
| + increment | 0.6229 | 0.1196 |
| + mover's rating | 0.6269 | 0.1229 |
| + opponent's rating | 0.6269 | 0.1226 |
| + material difference | 0.6961 | 0.1543 |
| + legal move count | 0.7097 | 0.1625 |
| + hanging pieces (white) | 0.7107 | 0.1627 |
| + hanging pieces (black) | 0.7124 | 0.1633 |
| No board, all of the 10 other features | 0.6886 | 0.1479 |

- Material difference gave the biggest jump, +0.069 ROC-AUC.
- Hand-built features alone beat the board alone. 0.6886 vs 0.6230.
- Suprisingly, the clock and time control added essentially nothing. I expected those features to contribute quite a lot given how time pressure is a classic cause of blunders.

# Limitations
- Blitz only, ratings between 1000 and 2000, and one slice of one month, which means that the games come from early in the month
- Only games with engine evaluations are included which is about 6% of all lichess games.
- I did not try CNNs or other features.

# Run the program yourself
```bash
git clone https://github.com/elnbr/chess-blunder-predictor.git
cd chess-blunder-predictor
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# download the first 1 GB of the January 2024 file
mkdir data
curl -r 0-1073741824 -o data/lichess_2024_01_partial.pgn.zst https://database.lichess.org/standard/lichess_db_standard_rated_2024-01.pgn.zst

# get the features for 20,000 games.
python src/explore.py

# train models
python src/train_lightgbm.py
python src/train_nn.py
```

# Repository layout
```
src/
  explore.py
  train_lightgbm.py
  train_nn.py

notebooks/
  exp_lightgbm.ipynb

models/
results/
data/
```

# Credits
Game data from the [Lichess open database](https://database.lichess.org/).
The [win probability formula](https://lichess.org/page/accuracy) comes from lichess.
