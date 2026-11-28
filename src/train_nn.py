import json
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupShuffleSplit
from copy import deepcopy
import torch
from torch import nn
import numpy as np
import optuna

device = "cuda" if torch.cuda.is_available() else "cpu"

class BlunderPredictor(nn.Module):

    def __init__(self, n_in: int, width=512, dropout=0.3):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_in, width),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(width, width // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(width // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)

        return logits.squeeze(1)
    

def prepare_data():
    data = np.load("data/moves.npz")

    X, y, groups = data["X"], data["y"], data["groups"]

    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_val_idx, test_idx = next(gss_test.split(X, y, groups))
    X_test, y_test = X[test_idx], y[test_idx]

    X_train_val, y_train_val, groups_train_val = X[train_val_idx], y[train_val_idx], groups[train_val_idx]

    gss_val = GroupShuffleSplit(n_splits=1, test_size=(0.15 / 0.85), random_state=42)
    train_idx, val_idx = next(gss_val.split(X_train_val, y_train_val, groups_train_val))

    X_train, y_train = X_train_val[train_idx], y_train_val[train_idx]
    X_val, y_val = X_train_val[val_idx], y_train_val[val_idx]

    # X_train has 764802 rows

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model(params, X_train, y_train, X_val, y_val, epochs):
    model = BlunderPredictor(X_train.shape[1], params["width"], params["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    bs = params["batch_size"]

    best_auc, best_state = 0.0, None

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(X_train), device=device)
        for i in range(0, len(order), bs):
            batch = order[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(X_train[batch]), y_train[batch])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val).cpu().numpy()
        auc = roc_auc_score(y_val.cpu().numpy(), val_logits)
        print(f"epoch {epoch + 1}: val AUC {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_state = deepcopy(model.state_dict())


    model.load_state_dict(best_state)
    return model, best_auc


def objective(trial):
    params = {
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "width": trial.suggest_categorical("width", [256, 512, 1024]),
        "batch_size": trial.suggest_categorical("batch_size", [512, 1024, 2048])
    }

    _, auc = train_model(params, X_train[:300_000], y_train[:300_000], X_val, y_val, epochs=8)

    return auc


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    X_train, y_train, X_val, y_val, X_test, y_test = [
        torch.as_tensor(a, dtype=torch.float32) for a in prepare_data()
    ]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=30)

    best = study.best_params
    print("best params:", best)

    with open("results/best_params.json", "w") as f:
        json.dump(best, f, indent=2)

    model, val_auc = train_model(best, X_train, y_train, X_val, y_val, epochs=30)

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_test)).cpu().numpy()
    y_true = y_test.cpu().numpy()

    print(f"val ROC-AUC:    {val_auc:.4f}")
    print(f"test ROC-AUC:   {roc_auc_score(y_true, probs):.4f}")
    print(f"test PR-AUC:    {average_precision_score(y_true, probs):.4f}")

    torch.save(model.state_dict(), "models/blunder_predictor_mlp.pt")
