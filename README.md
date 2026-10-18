# Bitcoin Next-Day Direction Predictor

A Streamlit app that uses historical Bitcoin prices and a random forest model
to predict whether the next day's closing price will increase.

## Setup and run

Run these commands from the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```

Open http://localhost:8501 in your browser. Press Control+C in the terminal to stop
the app.

For subsequent runs:

```bash
source .venv/bin/activate
python -m streamlit run app.py
```

The CSV in `data/` is the input dataset. Training creates `model.joblib`, which
is excluded from Git along with the virtual environment and Python caches.
Rerun `python train_model.py` after changing the training code or dataset.
