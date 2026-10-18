from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

from features import FEATURE_COLUMNS, add_features, load_data

df = add_features(load_data())
feature_columns = FEATURE_COLUMNS

model_data = df.dropna(subset=feature_columns + ["target"]).copy()
split_index = int(len(model_data) * 0.8)

train_data = model_data.iloc[:split_index].copy()
test_data = model_data.iloc[split_index:].copy()

X_train = train_data[feature_columns]
y_train = train_data["target"].astype(int)

X_test = test_data[feature_columns]
y_test = test_data["target"].astype(int)

print("\nTraining rows:", len(train_data))
print("Testing rows:", len(test_data))

print("\nTraining dates:")
print(train_data["datetime_utc"].min(), train_data["datetime_utc"].max())

print("\nTesting dates:")
print(test_data["datetime_utc"].min(), test_data["datetime_utc"].max())

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=10,
    random_state=42,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_predictions = baseline.predict(X_test)

print("\nRandom Forest accuracy:")
print(f"{accuracy_score(y_test, predictions):.2%}")

print("\nBaseline accuracy:")
print(f"{accuracy_score(y_test, baseline_predictions):.2%}")

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["DOWN/UNCHANGED", "UP"],
    zero_division=0,
))

print("\nConfusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

model_path = Path(__file__).parent / "model.joblib"
joblib.dump(
    {
        "model": model,
        "feature_columns": feature_columns,
        "test_accuracy": accuracy_score(y_test, predictions),
        "baseline_accuracy": accuracy_score(y_test, baseline_predictions),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]),
        "test_start": test_data["datetime_utc"].min(),
        "test_end": test_data["datetime_utc"].max(),
    },
    model_path,
)

print("\nSaved model to", model_path)