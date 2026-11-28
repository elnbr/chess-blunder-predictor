import json
from sklearn.metrics import roc_auc_score, average_precision_score
import optuna
import lightgbm as lgb
from lightgbm import LGBMClassifier
from train_nn import prepare_data


def score(model, X, y):
    probs = model.predict_proba(X)[:, 1]

    return roc_auc_score(y, probs), average_precision_score(y, probs)


def objective(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
    }

    model = LGBMClassifier(
        **params, subsample_freq=1, learning_rate=0.1, n_estimators=1000, metric="auc", verbose=-1
    )

    model.fit(
        X_train[:300_000], y_train[:300_000], 
        eval_X=X_val,
        eval_y=y_val,
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    return model.best_score_["valid_0"]["auc"]


if __name__ == "__main__":

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()

    default = LGBMClassifier(verbose=-1).fit(X_train, y_train)
    d_auc, d_ap = score(default, X_test, y_test)

    print(f"default LightGBM test ROC-AUC {d_auc:.4f}  PR-AUC {d_ap:.4f}")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=30)

    best = study.best_params
    print("best params:", best)


    final = LGBMClassifier(
        **best, subsample_freq=1, learning_rate=0.05, n_estimators=3000,
        metric="auc", verbose=-1
    )

    final.fit(
        X_train, y_train,
        eval_X=X_val,
        eval_y=y_val,
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )

    t_auc, t_ap = score(final, X_test, y_test)

    print(f"trees used: {final.best_iteration_}")
    print(f"tuned LightGBM test ROC-AUC {t_auc:.4f}    PR-AUC {t_ap:.4f}")

    results = {
        "default": {"test_roc_auc": d_auc, "test_pr_auc": d_ap},
        "tuned": {"test_roc_auc": t_auc, "test_pr_auc": t_ap, "params": best, "n_trees": final.best_iteration_}
    }

    with open("results/lgbm_results.json", "w") as f:
        json.dump(results, f, indent=2)
    final.booster_.save_model("models/lgbm_model.txt")
