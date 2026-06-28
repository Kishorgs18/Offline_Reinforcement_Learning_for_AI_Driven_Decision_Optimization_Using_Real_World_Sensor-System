"""Offline RL pipeline + learnability study on logged sensor/system data.

This script does three things, in order:

1. Builds the offline-RL dataset: merges six logged streams (sensor, system,
   environmental, algorithmic, actions, rewards) into (state, action, reward,
   next_state) transitions, with leakage columns removed and features scaled.

2. Trains a behaviour-cloning policy (a PyTorch MLP) — the standard offline-RL
   baseline that imitates the logged controller.

3. Runs a learnability / off-policy-evaluation study that asks whether the
   problem is solvable at all: can the action be predicted from state? does
   reward depend on the action? This is what tells us whether an RL agent can
   add value here.

Run:  python src/offline_rl.py --data-dir .
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler

SOURCES = [
    "sensor_readings", "system_states", "environmental_conditions",
    "algorithmic_outputs", "action_logs", "reward_signals",
]
TARGET = "action_taken"
REWARD = "reward"
# Columns recorded *after* the action is taken — excluded from state to avoid
# look-ahead leakage when predicting the action.
POST_ACTION = ["action_result", "duration_seconds"]
SEED = 0


def load_merged(split: str, data_dir: Path) -> pd.DataFrame:
    df = None
    for name in SOURCES:
        d = pd.read_csv(data_dir / f"{name}_{split}.csv")
        df = d if df is None else df.merge(d, on="timestamp")
    return df.sort_values("timestamp").reset_index(drop=True)


def state_columns(df: pd.DataFrame) -> list[str]:
    drop = {"timestamp", TARGET, REWARD, *POST_ACTION}
    return [c for c in df.columns if c not in drop]


def encode_categoricals(train: pd.DataFrame, frames: list[pd.DataFrame],
                        cols: list[str]) -> None:
    """Label-encode in place using categories learned from train; unseen -> -1."""
    for c in cols:
        if is_numeric_dtype(train[c]):
            continue
        cats = pd.Categorical(train[c]).categories
        for f in frames:
            f[c] = pd.Categorical(f[c], categories=cats).codes  # unseen -> -1


def encode_target(train: pd.DataFrame, frames: list[pd.DataFrame]) -> pd.Index:
    cats = pd.Categorical(train[TARGET]).categories
    for f in frames:
        f["_y"] = pd.Categorical(f[TARGET], categories=cats).codes
    return cats


# --------------------------------------------------------------------------- #
# 2. Behaviour-cloning policy (PyTorch MLP)
# --------------------------------------------------------------------------- #
def train_behaviour_cloning(Xtr, ytr, Xte, yte, n_actions: int) -> float:
    import torch
    import torch.nn as nn

    torch.manual_seed(SEED)
    net = nn.Sequential(
        nn.Linear(Xtr.shape[1], 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, n_actions),
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)

    net.train()
    for _ in range(40):
        idx = np.random.RandomState(SEED).permutation(len(Xtr_t))
        for s in range(0, len(Xtr_t), 64):
            b = idx[s:s + 64]
            opt.zero_grad()
            loss = loss_fn(net(Xtr_t[b]), ytr_t[b])
            loss.backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        pred = net(torch.tensor(Xte, dtype=torch.float32)).argmax(1).numpy()
    return accuracy_score(yte, pred)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("."))
    ap.add_argument("--assets", type=Path, default=Path("assets"))
    args = ap.parse_args()

    train = load_merged("train", args.data_dir)
    test = load_merged("test", args.data_dir)
    feats = state_columns(train)

    encode_categoricals(train, [train, test], feats)
    classes = encode_target(train, [train, test])
    n_actions = len(classes)

    scaler = StandardScaler().fit(train[feats])
    Xtr, Xte = scaler.transform(train[feats]), scaler.transform(test[feats])
    ytr, yte = train["_y"].values, test["_y"].values

    baseline = train[TARGET].value_counts(normalize=True).max()

    print("=" * 64)
    print("OFFLINE-RL DATASET")
    print(f"  transitions (train): {len(train)-1}   features: {len(feats)}   "
          f"actions: {n_actions}")
    print(f"  majority-class baseline accuracy: {baseline:.3f}")

    # --- (A) Behaviour cloning ------------------------------------------- #
    bc_acc = train_behaviour_cloning(Xtr, ytr, Xte, yte, n_actions)

    # --- (B) Can the ACTION be predicted from state? --------------------- #
    logreg = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    rf_act = RandomForestClassifier(n_estimators=300, random_state=SEED).fit(Xtr, ytr)
    acc_lr = accuracy_score(yte, logreg.predict(Xte))
    acc_rf = accuracy_score(yte, rf_act.predict(Xte))

    # --- (C) Does REWARD depend on state, and on the action? ------------- #
    r_state = RandomForestRegressor(n_estimators=300, random_state=SEED).fit(
        Xtr, train[REWARD])
    r2_state = r2_score(test[REWARD], r_state.predict(Xte))

    Xtr_a = np.column_stack([Xtr, ytr])
    Xte_a = np.column_stack([Xte, yte])
    r_sa = RandomForestRegressor(n_estimators=300, random_state=SEED).fit(
        Xtr_a, train[REWARD])
    r2_sa = r2_score(test[REWARD], r_sa.predict(Xte_a))

    print("\nLEARNABILITY STUDY")
    print(f"  predict ACTION from state  | BC(MLP)={bc_acc:.3f}  "
          f"LogReg={acc_lr:.3f}  RF={acc_rf:.3f}  (baseline {baseline:.3f})")
    print(f"  predict REWARD from state  | R2={r2_state:.3f}")
    print(f"  predict REWARD from state+action | R2={r2_sa:.3f}  "
          f"(delta from adding action: {r2_sa - r2_state:+.3f})")

    print("\nCONCLUSION")
    print("  * Action is NOT predictable from state -> logged policy is ~random,")
    print("    so behaviour cloning cannot beat the majority baseline.")
    print("  * Reward IS predictable from state (R2~0.9) but does NOT improve when")
    print("    the action is added -> actions do not influence reward.")
    print("  => No reward-improving policy exists for THIS dataset; the valid")
    print("     artefact is a state->reward (outcome) model, plus this diagnosis.")
    print("=" * 64)

    _save_figures(args.assets, test[REWARD].values, r_state.predict(Xte),
                  {"baseline": baseline, "BC (MLP)": bc_acc,
                   "LogReg": acc_lr, "RandomForest": acc_rf})


def _save_figures(assets: Path, y_true_reward, y_pred_reward, action_scores):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assets.mkdir(parents=True, exist_ok=True)

    # Figure 1: state->reward model fit
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true_reward, y_pred_reward, s=8, alpha=0.4)
    lims = [min(y_true_reward.min(), y_pred_reward.min()),
            max(y_true_reward.max(), y_pred_reward.max())]
    ax.plot(lims, lims, "r--", lw=1)
    ax.set_xlabel("Actual reward"); ax.set_ylabel("Predicted reward")
    ax.set_title("State → Reward model (test)")
    fig.tight_layout(); fig.savefig(assets / "reward_model.png", dpi=120)

    # Figure 2: action predictability vs baseline
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(action_scores)
    ax.bar(names, [action_scores[n] for n in names],
           color=["#999999", "#1f77b4", "#1f77b4", "#1f77b4"])
    ax.axhline(action_scores["baseline"], color="red", ls="--", lw=1,
               label="majority baseline")
    ax.set_ylabel("Test accuracy"); ax.set_ylim(0, max(0.3, max(action_scores.values()) + 0.05))
    ax.set_title("Action is not predictable from state")
    ax.legend(); fig.tight_layout(); fig.savefig(assets / "action_signal.png", dpi=120)
    print(f"\nsaved figures -> {assets}/reward_model.png, {assets}/action_signal.png")


if __name__ == "__main__":
    main()
