from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


RANDOM_STATE = 42


# -----------------------------
# Reddit: labeling
# -----------------------------
RISK_MAPPING: Dict[str, int] = {
    "suicidewatch": 1,
    "depression": 0,
    "anxiety": 0,
    "lonely": 0,
    "mentalhealth": 0,
}

LABEL_NAME: Dict[int, str] = {0: "Mental Health Risk", 1: "High Risk (Suicidal)"}


def apply_risk_labels(df: pd.DataFrame, subreddit_col: str = "subreddit") -> pd.DataFrame:
    if subreddit_col not in df.columns:
        raise KeyError(
            f"Expected a '{subreddit_col}' column but it was not found. "
            f"Available columns: {list(df.columns)}"
        )

    out = df.copy()
    out[subreddit_col] = out[subreddit_col].astype(str).str.strip().str.lower()
    out["risk_label"] = out[subreddit_col].map(RISK_MAPPING)
    out = out.dropna(subset=["risk_label"]).copy()
    out["risk_label"] = out["risk_label"].astype(int)
    out["risk_label_name"] = out["risk_label"].map(LABEL_NAME)
    return out


# -----------------------------
# Loading utilities
# -----------------------------
def load_csvs(
    data_path: str,
    file_glob: str = "**/*.csv",
    max_files: Optional[int] = None,
    encoding: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[Path]]:
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(
            f"DATA_PATH does not exist: {data_path}. "
            "Set DATA_PATH to your Kaggle CSV file, or to a directory containing CSVs."
        )

    if p.is_file():
        files = [p]
    else:
        files = sorted([f for f in p.glob(file_glob) if f.is_file() and f.suffix.lower() == ".csv"])

    if not files:
        raise FileNotFoundError(f"No CSV files found under: {p} (glob: {file_glob}).")

    if max_files is not None:
        files = files[:max_files]

    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, encoding=encoding))
        except UnicodeDecodeError:
            dfs.append(pd.read_csv(f, encoding=encoding or "latin-1"))
        except Exception as e:
            raise RuntimeError(f"Failed to read {f}: {e}")

    df = pd.concat(dfs, ignore_index=True)
    return df, files


def pick_text_column(df: pd.DataFrame, preferred: Iterable[str] = ("selftext", "body")) -> str:
    for c in preferred:
        if c in df.columns:
            return c
    raise KeyError(
        f"Could not find any expected text column {list(preferred)}. "
        f"Available columns: {list(df.columns)}"
    )


# -----------------------------
# Text preprocessing
# -----------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"\b(?:r|u)/[A-Za-z0-9_]+\b", flags=re.IGNORECASE)
ALLOWED_CHARS_RE = re.compile(r"[^a-z0-9\s\.,!?;:'\"\-\(\)]")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if text is None:
        return ""
    t = str(text).strip()
    if not t:
        return ""

    lowered = t.lower()
    if lowered in {"[deleted]", "[removed]"}:
        return ""

    lowered = URL_RE.sub(" ", lowered)
    lowered = MENTION_RE.sub(" ", lowered)
    lowered = ALLOWED_CHARS_RE.sub(" ", lowered)
    lowered = WHITESPACE_RE.sub(" ", lowered).strip()
    return lowered


def word_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def dedupe_by_author_most_recent(
    df: pd.DataFrame,
    author_col: str = "author",
    created_utc_col: str = "created_utc",
    fallback_time_col: str = "timestamp",
) -> pd.DataFrame:
    if author_col not in df.columns:
        return df

    out = df.copy()
    if created_utc_col in out.columns:
        out[created_utc_col] = pd.to_numeric(out[created_utc_col], errors="coerce")
        sort_col = created_utc_col
    elif fallback_time_col in out.columns:
        out[fallback_time_col] = pd.to_datetime(out[fallback_time_col], errors="coerce")
        sort_col = fallback_time_col
    else:
        out["__row_id"] = np.arange(len(out))
        sort_col = "__row_id"

    out = out.sort_values(sort_col, ascending=True)
    out = out.drop_duplicates(subset=[author_col], keep="last")
    if "__row_id" in out.columns:
        out = out.drop(columns=["__row_id"])
    return out


def balance_undersample(df: pd.DataFrame, label_col: str = "risk_label", target_per_class: int = 2000) -> Tuple[pd.DataFrame, int]:
    counts = df[label_col].value_counts().sort_index()
    min_count = int(counts.min())
    if min_count <= 0:
        raise ValueError("At least one class has 0 samples after preprocessing.")

    effective_target = min(target_per_class, min_count)
    balanced = (
        df.groupby(label_col, group_keys=False)
        .apply(lambda g: g.sample(n=effective_target, random_state=RANDOM_STATE, replace=False))
        .reset_index(drop=True)
    )
    return balanced, effective_target


# -----------------------------
# Embeddings
# -----------------------------
def generate_embeddings(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    normalize_embeddings: bool = False,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    return emb.astype(np.float32)


# -----------------------------
# Models + evaluation
# -----------------------------
def build_models() -> Dict[str, Pipeline]:
    return {
        "LR": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "SVM": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "MLP": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(256, 128),
                        max_iter=500,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def evaluate_models_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_dict: Dict[str, Pipeline],
    n_splits: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    cm_sum_by_model: Dict[str, np.ndarray] = {}
    cm_avg_by_model: Dict[str, np.ndarray] = {}

    labels_sorted = np.array(sorted(np.unique(y)))
    n_classes = len(labels_sorted)

    for model_name, model in model_dict.items():
        cm_sum = np.zeros((n_classes, n_classes), dtype=np.int64)
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            p_w, r_w, f1_w, _ = precision_recall_fscore_support(
                y_test, y_pred, average="weighted", zero_division=0
            )
            p_m, r_m, f1_m, _ = precision_recall_fscore_support(
                y_test, y_pred, average="macro", zero_division=0
            )

            cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
            cm_sum += cm

            rows.append(
                {
                    "model": model_name,
                    "fold": fold_idx,
                    "accuracy": acc,
                    "precision_weighted": p_w,
                    "recall_weighted": r_w,
                    "f1_weighted": f1_w,
                    "precision_macro": p_m,
                    "recall_macro": r_m,
                    "f1_macro": f1_m,
                }
            )

        cm_sum_by_model[model_name] = cm_sum
        cm_avg_by_model[model_name] = cm_sum / n_splits

    return pd.DataFrame(rows), cm_sum_by_model, cm_avg_by_model


def evaluate_transfer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_dict: Dict[str, Pipeline],
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    labels_sorted = np.array(sorted(np.unique(np.concatenate([y_train, y_test]))))
    rows = []
    cm_by_model: Dict[str, np.ndarray] = {}

    for model_name, model in model_dict.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )

        cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
        cm_by_model[model_name] = cm

        rows.append(
            {
                "model": model_name,
                "accuracy": acc,
                "precision_weighted": p_w,
                "recall_weighted": r_w,
                "f1_weighted": f1_w,
                "precision_macro": p_m,
                "recall_macro": r_m,
                "f1_macro": f1_m,
            }
        )

    return pd.DataFrame(rows).sort_values("f1_weighted", ascending=False), cm_by_model

