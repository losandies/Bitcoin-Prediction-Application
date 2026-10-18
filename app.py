from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from features import add_features, load_data

st.title("Bitcoin Next-Day Direction Predictor")

saved = joblib.load(Path(__file__).parent / "model.joblib")
model = saved["model"]
feature_columns = saved["feature_columns"]

df = add_features(load_data())
latest = df.tail(1)
latest_date = latest["datetime_utc"].iloc[0]
prediction_date = latest_date + pd.Timedelta(days=1)

up_index = list(model.classes_).index(1)
up_probability = model.predict_proba(latest[feature_columns])[0][up_index]
direction = "UP" if up_probability >= 0.5 else "DOWN/UNCHANGED"

st.subheader(f"Prediction for {prediction_date:%B %d, %Y}")
st.metric("Predicted direction", direction)
st.write(f"Estimated probability of UP: **{up_probability:.1%}**")
st.write(
    f"Based on the completed day {latest_date:%B %d, %Y}, "
    f"which closed at **\\${latest['close'].iloc[0]:,.2f}**."
)


st.subheader("Bitcoin Closing Price")
price_history = df.set_index("datetime_utc")["close"]
st.line_chart(price_history, y_label="Closing Price (USD)")

st.subheader("Historical Next-Day Direction")
direction_counts = df["target"].value_counts().rename(
    index={1: "UP", 0: "DOWN/UNCHANGED"}
)
st.bar_chart(direction_counts, y_label="Number of days")

up_share = direction_counts["UP"] / direction_counts.sum()
majority_label = "UP" if up_share >= 0.5 else "DOWN/UNCHANGED"
majority_share = max(up_share, 1 - up_share)

st.caption(
    f"Each bar counts past days by what happened the following day. "
    f"{up_share:.1%} of days were followed by an UP close. "
    f"Always guessing {majority_label} would be right {majority_share:.1%} of the time, "
    f"so the model is only useful if it beats that."
)

st.subheader("Model Performance on Test Data")
first_predicted = saved["test_start"] + pd.Timedelta(days=1)
last_predicted = saved["test_end"] + pd.Timedelta(days=1)

st.write(
    f"Tested by predicting each day from {first_predicted:%B %d, %Y} "
    f"to {last_predicted:%B %d, %Y}, a period the model never saw during training."
)

col1, col2 = st.columns(2)
col1.metric("Model accuracy", f"{saved['test_accuracy']:.2%}")
col2.metric(
    "Baseline accuracy",
    f"{saved['baseline_accuracy']:.2%}",
    help="Always guessing the most common direction from the training data.",
)

# Turn the 2x2 grid into one row per box so it can be charted.
labels = ["DOWN/UNCHANGED", "UP"]
matrix = pd.DataFrame(saved["confusion_matrix"], index=labels, columns=labels)
matrix_long = matrix.reset_index(names="Actual").melt(
    id_vars="Actual", var_name="Predicted", value_name="Days"
)

boxes = alt.Chart(matrix_long).mark_rect().encode(
    x=alt.X("Predicted:N", sort=labels, axis=alt.Axis(labelAngle=0)),
    y=alt.Y("Actual:N", sort=labels),
    color=alt.Color("Days:Q", scale=alt.Scale(scheme="blues"), legend=None),
    tooltip=["Actual", "Predicted", "Days"],
)

# Show the number in each box. Use white text on the dark boxes.
numbers = boxes.mark_text(fontSize=20).encode(
    text="Days:Q",
    color=alt.condition(
        alt.datum.Days > matrix_long["Days"].mean(),
        alt.value("white"),
        alt.value("black"),
    ),
)

st.write("**Confusion matrix:** how the model's predictions compared with what actually happened.")
st.altair_chart((boxes + numbers).properties(height=300), width="stretch")

st.subheader("Check a Past Prediction")

usable = df.dropna(subset=feature_columns + ["target"]).copy()
# Each row predicts the next day, so label it with that day.
usable["predicted_day"] = (usable["datetime_utc"] + pd.Timedelta(days=1)).dt.date

chosen_date = st.date_input(
    "Pick a day to see what the model predicted for it:",
    value=usable["predicted_day"].max(),
    min_value=usable["predicted_day"].min(),
    max_value=usable["predicted_day"].max(),
)

row = usable[usable["predicted_day"] == chosen_date]

if row.empty:
    st.warning(
        "No prediction available for that day. That day or the day before it "
        "is either missing from the data or had no trading (zero volume)."
    )
else:
    past_up_probability = model.predict_proba(row[feature_columns])[0][up_index]
    predicted = "UP" if past_up_probability >= 0.5 else "DOWN/UNCHANGED"
    actual = "UP" if row["target"].iloc[0] == 1 else "DOWN/UNCHANGED"

    st.write(
        f"Prediction made using data from {row['datetime_utc'].iloc[0]:%B %d, %Y}, "
        f"which closed at **\\${row['close'].iloc[0]:,.2f}**. "
        f"{chosen_date:%B %d, %Y} closed at **\\${row['next_close'].iloc[0]:,.2f}**."
    )

    col1, col2 = st.columns(2)
    col1.metric("Model predicted", predicted, help=f"Probability of UP: {past_up_probability:.1%}")
    col2.metric("Actually happened", actual)

    if predicted == actual:
        st.success("The model was correct for this day.")
    else:
        st.error("The model was wrong for this day.")

    if row["datetime_utc"].iloc[0] < saved["test_start"]:
        st.info(
            "This day is in the training period. The model learned from it, "
            "so this result doesn't show how it performs on unseen data."
        )
