import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "is_canceled"
LEAKAGE_COLUMNS = ["reservation_status", "reservation_status_date"]

MONTH_TO_NUMBER = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Выполняет базовую очистку, удаляет некорректные строки и leakage-колонки"""
    cleaned = df.copy()

    cleaned["children"] = cleaned["children"].fillna(0)
    cleaned["country"] = cleaned["country"].fillna("Unknown")
    cleaned["agent"] = cleaned["agent"].fillna(0)
    cleaned["company"] = cleaned["company"].fillna(0)

    cleaned = cleaned.drop_duplicates()

    guests = cleaned["adults"] + cleaned["children"] + cleaned["babies"]
    nights = cleaned["stays_in_weekend_nights"] + cleaned["stays_in_week_nights"]
    cleaned = cleaned[(guests > 0) & (nights > 0)]
    cleaned = cleaned[cleaned["adr"] >= 0]

    cleaned["children"] = cleaned["children"].astype(int)
    cleaned["agent"] = cleaned["agent"].astype(int)
    cleaned["company"] = cleaned["company"].astype(int)

    cleaned = cleaned.drop(columns=LEAKAGE_COLUMNS, errors="ignore")

    return cleaned.reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Создаёт новые признаки для обучения моделей"""
    featured = df.copy()

    featured["arrival_month_num"] = featured["arrival_date_month"].map(MONTH_TO_NUMBER).astype(int)
    featured["total_guests"] = featured["adults"] + featured["children"] + featured["babies"]
    featured["total_nights"] = featured["stays_in_weekend_nights"] + featured["stays_in_week_nights"]
    featured["has_children"] = ((featured["children"] + featured["babies"]) > 0).astype(int)
    featured["has_agent"] = (featured["agent"] > 0).astype(int)
    featured["has_company"] = (featured["company"] > 0).astype(int)
    featured["adr_per_person"] = featured["adr"] / featured["total_guests"].clip(lower=1)
    featured["room_type_changed"] = (featured["reserved_room_type"] != featured["assigned_room_type"]).astype(int)

    return featured


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Подготавливает данные и удаляет признаки с утечкой таргета"""
    prepared = add_features(clean_data(df))
    prepared = prepared.drop(columns=LEAKAGE_COLUMNS, errors="ignore")
    return prepared.drop_duplicates().reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Разделяет признаки на числовые и категориальные"""
    features = df.drop(columns=[TARGET_COLUMN])
    numeric_features = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = features.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    return numeric_features, categorical_features


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Собирает общий sklearn-препроцессинг для признаков"""
    numeric_features, categorical_features = get_feature_columns(df)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
