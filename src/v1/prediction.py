import sys
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from src.v1.config import PROJECT_ROOT, ACTIVE, PIPELINE_PATH
from src.v1.main import run_prediction

# adding src/ to the path dynamically so internal module imports execute smoothly
sys.path.append(str(PROJECT_ROOT))

# for single inference
def predict_freight_rate(payload: dict) -> dict:
    
    active_model_json_path = Path(ACTIVE)
    
    # checking if active_model exists
    if not active_model_json_path.exists():
        raise FileNotFoundError(
            f"Active model metadata file not found at {active_model_json_path}. "
            "Please run train_pipeline first to get production model"
        )
    
    # extract the production model name and version
    with open(active_model_json_path, "r") as f:
        active_meta = json.load(f)
        
    model_name = active_meta["model_name"]
    model_version = active_meta["version"]
    
    print(f"Loaded active production metadata: {model_name} (Version {model_version})")
        
    # load the compiled binary estimator object (.pkl) from the version subdirectory
    model_file_path = PROJECT_ROOT / f"models/models/v{model_version}/{model_name}_best_{model_version}.pkl"
    
    if not model_file_path.exists():
        raise FileNotFoundError(f"Serialized model estimator binary file not found at {model_file_path}.")
    
    fitted_model = joblib.load(model_file_path)
    
    # package the active model configuration tuple instance parameters
    active_model_instance = (fitted_model, model_version, model_name)
    
    # execute inference via the runtime prediction loop orchestrated in main.py
    print("Routing shipment payload criteria through the preprocessing and forecasting engine...")
    
    prediction_result = run_prediction(payload, active_model_instance)
    
    return prediction_result


# for batch inference
def run_batch_validation():
    validation_data_path = (
        PROJECT_ROOT / "data/raw/validation.csv"
    )
    template_path = (
        PROJECT_ROOT / "data/raw/validation-predictions-template.csv"
    )
    output_path = PROJECT_ROOT / "validation_predictions.csv"
    active_path = Path(ACTIVE)

    if not validation_data_path.exists() or not template_path.exists():
        raise FileNotFoundError(
            "Required validation.csv or validation-predictions-template.csv "
            "is missing from data/raw/."
        )

    if not active_path.exists():
        raise FileNotFoundError(
            f"Active model metadata not found: {active_path}"
        )

    df_val = pd.read_csv(validation_data_path)
    df_template = pd.read_csv(template_path)

    # handling for load_id as it is necessary to move forward
    if "load_id" not in df_val or "load_id" not in df_template:
        raise KeyError("Both files must contain a load_id column.")

    print(f"Loaded {len(df_val)} validation records.")

    # load the active model once.
    with open(active_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model_name = metadata["model_name"]
    model_version = metadata["version"]

    model_path = (
        PROJECT_ROOT
        / f"models/models/v{model_version}/"
          f"{model_name}_best_{model_version}.pkl"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    fitted_model = joblib.load(model_path)
    model_instance = (
        fitted_model,
        model_version,
        model_name,
    )

    # call run_prediction for the entire dataset.
    result = run_prediction(df_val, model_instance)
    predicted_rates = result["predicted_rate_usd"]

    if len(predicted_rates) != len(df_val):
        raise ValueError(
            f"Expected {len(df_val)} predictions, "
            f"received {len(predicted_rates)}."
        )

    # align predictions with the template by load_id.
    if not df_val["load_id"].reset_index(drop=True).equals(
        df_template["load_id"].reset_index(drop=True)
    ):
        prediction_map = dict(
            zip(df_val["load_id"], predicted_rates)
        )
        df_template["predicted_rate"] = (
            df_template["load_id"].map(prediction_map)
        )
    else:
        df_template["predicted_rate"] = predicted_rates

    # handle missing predictions if any.
    rates = pd.to_numeric(
        df_template["predicted_rate"], errors="coerce"
    )
    valid_rates = rates[np.isfinite(rates)]

    # for NaN predictions
    if rates.isna().any():
        if valid_rates.empty:
            raise ValueError("No valid predictions available.")
        df_template["predicted_rate"] = rates.fillna(
            valid_rates.median()
        )

    # outputing the required validation_predictions.csv
    df_template["predicted_rate"] = (
        df_template["predicted_rate"].round(2)
    )

    df_template.to_csv(output_path, index=False)

    print(f"Predictions saved to: {output_path}")
    print(df_template.head(10).to_string())
    

# for december-chart-inputs.csv
def run_batch_validation_dec():
    input_path = PROJECT_ROOT / "data/raw/december-chart-inputs.csv"
    output_path = input_path  # currently it is designed to overwrite the same CSV (in future it can be changed as necessary)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    active_path = Path(ACTIVE)
    pipeline_path = Path(PIPELINE_PATH)

    if not active_path.exists():
        raise FileNotFoundError(
            f"Active model metadata not found: {active_path}"
        )

    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Fitted preprocessing pipeline not found: {pipeline_path}"
        )

    # read the original CSV and preserve its exact schema.
    df = pd.read_csv(input_path)
    original_columns = df.columns.tolist()

    # this is columns as per the file
    required_input_columns = [
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date",
        "predicted_rate",
    ]

    # finding the missing columns
    missing_input = [
        col for col in required_input_columns
        if col not in df.columns
    ]

    if missing_input:
        raise ValueError(
            f"December input is missing required columns: {missing_input}"
        )

    if df.empty:
        raise ValueError("December input CSV contains no rows.")

    # the raw input columns used during training.
    fitted_pipeline = joblib.load(pipeline_path)

    if not hasattr(fitted_pipeline, "feature_names_in_"):
        raise ValueError(
            "The saved preprocessing pipeline does not expose "
            "feature_names_in_. Cannot safely reconstruct its input schema."
        )

    expected_columns = list(fitted_pipeline.feature_names_in_)

    # columns to be added as it is missing in the file
    feature_engineering_columns = {
        "weight",
        "equipment",
        "market_index",
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "distance",
        "quote_signal",
    }

    missing_from_pipeline = (
        feature_engineering_columns - set(expected_columns)
    )

    if missing_from_pipeline:
        raise ValueError(
            "The saved pipeline schema does not contain columns required "
            "by CustomFreightFeatureEngineer: "
            f"{sorted(missing_from_pipeline)}"
        )

    # prepare a temporary prediction frame
    # exclude the existing prediction so it cannot be used as a feature
    prediction_input = df.drop(
        columns=["predicted_rate"],
        errors="ignore"
    ).copy()

    # add missing training features as NaN.
    # the preprocessing pipeline will impute these values.
    missing_features = [
        col for col in expected_columns
        if col not in prediction_input.columns
    ]

    for col in missing_features:
        prediction_input[col] = np.nan

    # keep only the original training features, in training order
    # this prevents the December output column from entering the model
    prediction_input = prediction_input.reindex(
        columns=expected_columns
    )

    print(f"Loaded {len(prediction_input)} December loads.")
    print(f"Added missing features as NaN: {missing_features}")

    # load the active model
    with open(active_path, "r") as f:
        active_meta = json.load(f)

    model_name = active_meta["model_name"]
    model_version = active_meta["version"]

    model_path = (
        PROJECT_ROOT
        / f"models/models/v{model_version}/{model_name}_best_{model_version}.pkl"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Serialized model not found: {model_path}"
        )

    fitted_model = joblib.load(model_path)
    model_instance = (fitted_model, model_version, model_name)

    # run_prediction() calls preprocessor(..., fit_transform="no"),
    # which loads the fitted pipeline and performs imputation,
    # feature engineering, and encoding before predicting.
    result = run_prediction(prediction_input, model_instance)

    predicted_rates = np.asarray(
        result["predicted_rate_usd"],
        dtype=float
    ).reshape(-1)

    if len(predicted_rates) != len(df):
        raise ValueError(
            f"Expected {len(df)} predictions, got {len(predicted_rates)}."
        )

    if not np.isfinite(predicted_rates).all():
        raise ValueError(
            "Predictions contain NaN or infinite values; "
            "the original CSV has not been overwritten."
        )

    # update only the prediction column.
    df["predicted_rate"] = np.round(predicted_rates, 2)

    # guarantee the original column names and order.
    df = df[original_columns]

    # overwrite the original December input file.
    df.to_csv(output_path, index=False)

    print(f"[SUCCESS] Predictions saved to: {output_path}")
    print(
        df[["pickup", "delivery", "predicted_rate"]]
        .head(10)
        .to_string(index=False)
    )

    return {
        "status": "success",
        "rows_processed": len(df),
        "output_path": str(output_path),
        "model_name": model_name,
        "model_version": model_version,
        "added_missing_features": missing_features,
    }