import pandas as pd

def extract_features_from_column(series: pd.Series):
    return {
        'dtype_encoded': {'object': 0, 'int64': 1, 'float64': 2}.get(str(series.dtype), 3),
        'null_pct': series.isnull().mean(),
        'unique_pct': series.nunique() / len(series),
        'avg_len': series.astype(str).str.len().mean() if series.dtype == 'object' else 0
    }

def predict_column_transformation(col_series, model, label_encoder):
    features = extract_features_from_column(col_series)
    X_input = pd.DataFrame([features])
    pred_encoded = model.predict(X_input)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]
    return pred_label
