from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import TARGET_COLUMN, build_preprocessor

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "hotel_bookings.csv"


def load_raw_data(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Загружает исходный датасет бронирований с диска"""
    return pd.read_csv(path)


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Разбивает данные на train/validation/test со стратификацией"""
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df[TARGET_COLUMN],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def evaluate_classifier(model, x_val: pd.DataFrame, y_val: pd.Series) -> dict[str, float]:
    """Считает основные метрики на validation-выборке"""
    predictions = model.predict(x_val)

    metrics = {
        "accuracy": accuracy_score(y_val, predictions),
        "precision": precision_score(y_val, predictions, zero_division=0),
        "recall": recall_score(y_val, predictions, zero_division=0),
        "f1": f1_score(y_val, predictions, zero_division=0),
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_val)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_val, probabilities)
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


def build_model_pipeline(train_df: pd.DataFrame, estimator) -> Pipeline:
    """Объединяет препроцессинг и модель в один sklearn-pipeline"""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(train_df)),
            ("model", estimator),
        ]
    )
