from sklearn.dummy import DummyClassifier

from src.modeling import build_model_pipeline, load_raw_data, split_data
from src.preprocessing import LEAKAGE_COLUMNS, TARGET_COLUMN, prepare_data


def test_prepare_data_removes_leakage_columns_and_invalid_rows():
    raw_df = load_raw_data()
    prepared_df = prepare_data(raw_df)

    assert TARGET_COLUMN in prepared_df.columns
    assert not set(LEAKAGE_COLUMNS).intersection(prepared_df.columns)
    assert prepared_df.duplicated().sum() == 0
    assert (prepared_df["total_guests"] > 0).all()
    assert (prepared_df["total_nights"] > 0).all()
    assert (prepared_df["adr"] >= 0).all()


def test_split_data_keeps_all_parts_non_empty_and_stratified():
    prepared_df = prepare_data(load_raw_data())
    train_df, val_df, test_df = split_data(prepared_df)

    assert len(train_df) > len(val_df) > 0
    assert len(train_df) > len(test_df) > 0
    assert len(train_df) + len(val_df) + len(test_df) == len(prepared_df)

    original_rate = prepared_df[TARGET_COLUMN].mean()
    for part in [train_df, val_df, test_df]:
        assert abs(part[TARGET_COLUMN].mean() - original_rate) < 0.01


def test_model_pipeline_can_fit_and_predict_on_small_sample():
    prepared_df = prepare_data(load_raw_data()).sample(n=500, random_state=42)
    train_df, val_df, _ = split_data(prepared_df)

    model = build_model_pipeline(train_df, estimator=DummyClassifier(strategy="most_frequent"))

    x_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    x_val = val_df.drop(columns=[TARGET_COLUMN])

    model.fit(x_train, y_train)
    predictions = model.predict(x_val)

    assert len(predictions) == len(x_val)
