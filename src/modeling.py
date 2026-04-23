from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import TARGET_COLUMN, build_preprocessor, prepare_data

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "hotel_bookings.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
VAL_DATA_PATH = PROCESSED_DATA_DIR / "val.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"
METRICS_PATH = PROCESSED_DATA_DIR / "metrics_cp1.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model_cp1.joblib"


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
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


def get_models() -> dict[str, object]:
    """Возвращает набор моделей для сравнения"""
    return {
        "knn_classifier": KNeighborsClassifier(n_neighbors=15),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=14,
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def build_model_pipeline(train_df: pd.DataFrame, estimator) -> Pipeline:
    """Объединяет препроцессинг и модель в один sklearn-pipeline"""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(train_df)),
            ("model", estimator),
        ]
    )


def train_and_evaluate_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[pd.DataFrame, Pipeline]:
    """Обучает все модели и выбирает лучшую по F1-score"""
    x_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    x_val = val_df.drop(columns=[TARGET_COLUMN])
    y_val = val_df[TARGET_COLUMN]

    rows = []
    fitted_models = {}

    for model_name, estimator in get_models().items():
        pipeline = build_model_pipeline(train_df, estimator)
        pipeline.fit(x_train, y_train)
        metrics = evaluate_classifier(pipeline, x_val, y_val)
        rows.append({"model": model_name, **metrics})
        fitted_models[model_name] = pipeline

    metrics_df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    best_model_name = metrics_df.loc[0, "model"]

    return metrics_df, fitted_models[best_model_name]


def run_training() -> pd.DataFrame:
    """Запускает полный training workflow и сохраняет метрики и лучшую модель"""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    prepared_df = prepare_data(load_raw_data())
    train_df, val_df, test_df = split_data(prepared_df)

    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    val_df.to_csv(VAL_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    metrics_df, best_model = train_and_evaluate_models(train_df, val_df)
    metrics_df.to_csv(METRICS_PATH, index=False)
    dump(best_model, MODEL_PATH)

    return metrics_df


if __name__ == "__main__":
    print(run_training().to_string(index=False))
