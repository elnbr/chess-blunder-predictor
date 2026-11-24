# chess-blunder-predictor
Predicts the probability that a chess player blunders on their next move, using the position, ratings, side to move, and clock time.

# Data
- **Source:** Lichess open database, January 2024, first 1 GB of the file.
- **Filters:** rated blitz game, standard chess, both players rated between 1000 and 2000, engine evaluations and clock times present.
- **Size:** 20,000 games, resulting in 1,090,155 moves total. Skipped the first 5 moves on each game because its mostly opening theory.
- **Blunder rate:** 8.1% of all moves.

# How a blunder is defined
A blunder is a move that results in a 15 or more percentage drop of win probability for the mover.

# Features (total of 778 features)
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
- **Split:**
- **Metrics:**
- **Models:**

# Results:
| **Model** | **ROC-AUC** | **PR-AUC** |
| --------- | ----------- | ---------- |
| Random Guessing | 0.500  | 0.081      |
| LightGBM (default) | 0.712 | 0.163    |
| Neural network (tuned) | NaN | NaN |

# What each feature group adds to result (LightGBM default settings)
| **Features used** | **ROC-AUC** | **PR-AUC** |
| ----------------- | ----------- | ---------- |
| Side to move, clock features, ratings, material difference, number of legal moves, and number of hanging pieces | 0.689 | 0.148 |
| Board only | 0.623 | 0.120 |
| Everything | 0.712 | 0.163 |
