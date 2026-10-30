import json
import pandas as pd
import numpy as np
from pathlib import Path
from src.v1.preprocessing import load_data, preprocessor
from src.v1.train import trainer
from src.v1.evaluate import evaluator
from src.v1.config import (
    VERSION, ACTIVE, MODELS, PARAMS, PROJECT_ROOT
)


# independant function for handling path resolvement which is unnecessary but this way it is easier
def get_resolved_path(path_str: str) -> Path:
    path_obj = Path(path_str)
    return path_obj if path_obj.is_absolute() else (PROJECT_ROOT / path_str).resolve()


# MAIN TRAINING PIPELINE
def train_pipeline(ver=VERSION):
    
    print(f"Starting the training pipeline with version: {ver}")
    
    # loading the data
    Xtr, Xte, ytr, yte = load_data(ver=ver)

    # preprocessing
    X_train_processed, X_train_std = preprocessor(Xtr, fit_transform="yes", std="yes", ver=ver)
    X_test_processed, X_test_std = preprocessor(Xte, fit_transform="no", std="yes", ver=ver)

    # model training
    train_output = trainer(X_train_processed, X_train_std, ytr, MODELS, PARAMS, 'r2', ver)
    
    if isinstance(train_output, tuple):
        models, _ = train_output
    else:
        models = train_output

    # evaluation of the models
    best_model_name, best_info, _ = evaluator(X_test_processed, X_test_std, yte, models, plot='yes', ver=ver)
    
    test_metrics = best_info['test_metrics']

    # metrics
    result = {
        "version": int(ver),
        "model_name": best_model_name,
        "test_r2_score": float(test_metrics["r2"]),
        "test_rmse_dollars": float(test_metrics["rmse"]),
        "test_mae_dollars": float(test_metrics["mae"]),
        "val_r2_score": float(best_info["val_metrics"]["r2"]),
        "val_rmse_dollars": float(best_info["val_metrics"]["rmse"])
    }

    # checking if the model is performing as required
    if float(result["test_r2_score"]) < 0.70:
        
        print(f"\nModel rejected: R² ({result['test_r2_score']:.4f}) < 0.70 minimum threshold floor")
        
        return {"status": "model performance did not cross the required criteria threshold", **result}

    print(f"\nModel promoted: R² ({result['test_r2_score']:.4f}) >= 0.70 minimum threshold floor")

    # for saving the required active model metadata configuration file to disk
    active_path = get_resolved_path(ACTIVE)
    active_path.parent.mkdir(parents=True, exist_ok=True)

    with open(active_path, "w") as f:
        json.dump(result, f, indent=2)

    return {"status": "model is completely trained and functional", **result}



def run_prediction(data, m_inst):

    model, ver_val, name = m_inst
    m_low = name.lower()

    is_scaled_model = any(
        key in m_low
        for key in [
            "linearregression",
            "ridge",
            "lasso",
            "kneighbor",
            "svr",
            "logisticregression",
        ]
    )

    # accept one shipment or an entire DataFrame.
    is_single_record = isinstance(data, dict)

    # converting the raw dataset into dataframe if not already
    if is_single_record:
        data_copy = pd.DataFrame([data])
    elif isinstance(data, pd.DataFrame):
        data_copy = data.copy()
    else:
        raise TypeError(
            "data must be a shipment dictionary or a pandas DataFrame."
        )

    # handling empty dataset
    if data_copy.empty:
        raise ValueError("Cannot predict an empty dataset.")

    # apply the fitted preprocessing.
    if is_scaled_model:
        _, eval_data = preprocessor(
            data_copy, "no", "yes", ver_val
        )
    else:
        eval_data = preprocessor(
            data_copy, "no", "no", ver_val
        )

    # predict all rows in one call.
    predicted_log_rates = np.asarray(
        model.predict(eval_data)
    ).reshape(-1)

    # reverse log1p transformation and prevent negative rates.
    predicted_rates = np.maximum(
        0.0, np.expm1(predicted_log_rates)
    )
    predicted_rates = np.round(predicted_rates, 2)

    if len(predicted_rates) != len(data_copy):
        raise ValueError(
            f"Expected {len(data_copy)} predictions, "
            f"but received {len(predicted_rates)}."
        )

    # preserve compatibility with individual predictions.
    output = (
        float(predicted_rates[0])
        if is_single_record
        else predicted_rates.tolist()
    )

    return {
        "predicted_rate_usd": output,
        "model_version": ver_val,
        "model_engine": name,
    }


if __name__ == '__main__':
    # training pipeline
    pipeline_status = train_pipeline(ver=VERSION)
    
    print("\nTraining complete\n")
    
    print(pipeline_status)