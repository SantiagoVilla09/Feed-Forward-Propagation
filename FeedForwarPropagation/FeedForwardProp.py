# feed_forward_from_csv.py
# Pure NumPy + Pandas feed-forward propagation with CSV support.
# No deep-learning frameworks are used (no keras/tf/sklearn).
# Comments are in English as requested.

# FeedForwardProp.py
# Pure NumPy + Pandas feed-forward propagation for your Student_Performance.csv
# No deep-learning frameworks used. Always reads Student_Performance.csv
# Comments are in English

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# ---------- Activation functions ----------
def sigmoid(z): 
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def tanh(z): return np.tanh(z)
def relu(z): return np.maximum(0, z)
def leaky_relu(z, alpha=0.01): return np.where(z > 0, z, alpha * z)
def linear(z): return z

def softmax(z, axis=1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)

ACTIVATIONS = {
    "sigmoid": sigmoid,
    "tanh": tanh,
    "relu": relu,
    "leaky_relu": leaky_relu,
    "linear": linear,
    "softmax": softmax
}

# ---------- Core feed-forward ----------
def feed_forward(layers: List[Dict], X: np.ndarray, return_cache=True):
    A = X
    cache = []
    for idx, layer in enumerate(layers):
        W, b = layer["W"], layer["b"]
        act = layer["activation"]
        Z = A @ W + b
        if act == "leaky_relu":
            A = leaky_relu(Z, layer.get("alpha", 0.01))
        else:
            A = ACTIVATIONS[act](Z)
        if return_cache:
            cache.append({"Z": Z, "A": A, "activation": act})
    return A, cache

# ---------- Utilities ----------
def he_init(n_in, n_out, rng):
    return rng.normal(0, np.sqrt(2.0 / n_in), size=(n_in, n_out))

def xavier_init(n_in, n_out, rng):
    return rng.normal(0, np.sqrt(1.0 / n_in), size=(n_in, n_out))

def build_layers(input_dim, hidden, last_activation, output_dim, seed=42):
    rng = np.random.default_rng(seed)
    sizes = [input_dim] + hidden + [output_dim]
    acts = ["relu"] * len(hidden) + [last_activation]
    layers = []
    for i in range(len(sizes) - 1):
        n_in, n_out = sizes[i], sizes[i + 1]
        act = acts[i]
        W = he_init(n_in, n_out, rng) if act in ("relu", "leaky_relu") else xavier_init(n_in, n_out, rng)
        b = np.zeros((1, n_out))
        layers.append({"W": W, "b": b, "activation": act})
    return layers

def preprocess_csv(csv_path: Path, drop_cols: Optional[List[str]] = None, standardize_numeric=False):
    df = pd.read_csv(csv_path)
    df_original = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(float).fillna(df[col].median())
        else:
            df[col] = df[col].astype(str).fillna("Unknown")
    if drop_cols:
        for c in drop_cols:
            if c in df.columns:
                df = df.drop(columns=c)
    X_df = pd.get_dummies(df, drop_first=False)
    if standardize_numeric:
        for c in X_df.columns:
            if pd.api.types.is_numeric_dtype(X_df[c]):
                std = X_df[c].std(ddof=0)
                if std > 0:
                    X_df[c] = (X_df[c] - X_df[c].mean()) / std
    return df_original, X_df

def append_and_save_predictions(df_original, y_pred, out_path: Path):
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    pred_df = pd.DataFrame(y_pred, columns=[f"nn_output_{i}" for i in range(y_pred.shape[1])])
    result = pd.concat([df_original.reset_index(drop=True), pred_df], axis=1)
    result.to_csv(out_path, index=False)
    return result

# ---------- MAIN ----------
def main():
    # Fixed file names
    csv_path = Path("Student_Performance.csv")
    out_path = Path("predictions.csv")

    # Load data
    df_original, X_df = preprocess_csv(csv_path)
    X = X_df.values.astype(float)

    # Define network: 5 inputs -> 16 -> 8 -> 1 output
    layers = build_layers(input_dim=X.shape[1], hidden=[16, 8],
                          last_activation="sigmoid", output_dim=1, seed=42)

    # Run feed-forward
    y_pred, cache = feed_forward(layers, X)

    # Save results
    result = append_and_save_predictions(df_original, y_pred, out_path)

    # Print network summary
    print("=== Feed-Forward Propagation Complete ===")
    print(f"Input file: {csv_path}")
    print(f"Output file: {out_path}")
    print(f"Network structure: {[X.shape[1]] + [l['W'].shape[1] for l in layers]}")
    total_neurons = X.shape[1] + sum(l['W'].shape[1] for l in layers)
    print(f"Total neurons (input+hidden+output): {total_neurons}\n")
    print(result.head())

if __name__ == "__main__":
    main()