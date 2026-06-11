# Architecture

Simple text diagram:

```text
User
  |
  v
Streamlit App (app.py)
  |
  v
Prediction Helpers (src/predict.py)
  |
  +--> Preprocessing (src/data_utils.py)
  |
  +--> Model Definition / Loading (src/model.py)
  |
  +--> Explainability Logic (src/explainability.py)
  |
  v
Prediction + Explanation Output
```

Future addition:

```text
Dataset -> Training Script -> Saved Model -> Streamlit App
```

