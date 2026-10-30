import sys
import streamlit as st
import pandas as pd
import json
import joblib
from pathlib import Path

from src.v1.config import PROJECT_ROOT, ACTIVE, VERSION
from src.v1.preprocessing import load_data, preprocessor
from src.v1.main import train_pipeline, run_prediction
from src.v1.prediction import predict_freight_rate


# simple app for this assignment
st.set_page_config(
    page_title="Freight rate prediction & MLOps orchestrator",
    layout="wide"
)

# choosing the function to be performed
st.title("Freight rate prediction & MLOps dashboard")
st.markdown("Orchestrate preprocessing, hyperparameter training, continuous evaluation, and live inference.")

st.sidebar.title("Navigation")

app_mode = st.sidebar.selectbox(
    "Choose action",
    [
        "Model training & evaluation",
        "Single load inference",
        "Batch validation pipeline",
        "December chart batch prediction"
    ]
)

# check whether the active model metadata file already exists on disk
active_path = Path(ACTIVE)
is_model_trained = active_path.exists()

# for starting training pipeline
if app_mode == "Model training & evaluation":
    st.header("Model training pipeline")
    st.markdown("Trigger the training pipeline using `train_pipeline()`. This will load raw data, run feature engineering, tune hyperparameters via cross-validation, evaluate models, and save artifacts.")

    version_input = st.sidebar.number_input("Model version", min_value=1, value=VERSION, step=1)

    # disable button if the model is already trained and active metadata exists
    train_clicked = st.button("Start training pipeline", type="primary", disabled=is_model_trained)

    if is_model_trained:
        st.info("Model has already been trained and deployed. The training pipeline is locked. To retrain, delete the `active_model.json` file or model artifacts directory.")

    if train_clicked:
        with st.status("Training models and optimizing thresholds... Please wait.", expanded=True) as status:
            try:
                result = train_pipeline(ver=version_input)
                status.update(label="Training pipeline completed successfully!", state="complete", expanded=False)
                st.success("Training pipeline completed successfully!")
                st.json(result)
                # force a rerun so the button instantly disables itself post-training
                st.rerun()
            except Exception as e:
                status.update(label="Training failed!", state="error")
                st.error(f"Training failed with error: {str(e)}")

    st.divider()
    st.subheader("Active model status & metrics")
    
    if active_path.exists():
        try:
            with open(active_path, "r") as f:
                content = f.read().strip()
                active_meta = json.loads(content) if content else {}
            
            if isinstance(active_meta, dict) and active_meta.get("model_name"):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Active model", active_meta.get("model_name"))
                col2.metric("Version", active_meta.get("version"))
                col3.metric("Test R² score", f"{active_meta.get('test_r2_score', 0):.4f}")
                col4.metric("Test RMSE", f"${active_meta.get('test_rmse_dollars', 0):,.2f}")
                
                eval_dir = PROJECT_ROOT / f"plots/evaluate/v{active_meta.get('version')}"
                
                if eval_dir.exists():
                    st.markdown("### Evaluation Visualizations")
                    png_files = list(eval_dir.glob("*.png"))
                    for img_path in png_files:
                        st.image(str(img_path), caption=img_path.name, width="stretch")
            else:
                st.info("Training metadata is initializing or empty. Please run the training pipeline.")
        except json.JSONDecodeError:
            st.warning("Metadata file is updating. Please refresh once training completes.")
    else:
        st.info("No active model metadata found. Please run the training pipeline first.")

# for single prediction pipeline
elif app_mode == "Single load inference":
    st.header("Real-time freight rate prediction")
    st.markdown("Input shipment specifications to calculate predicted freight rates in USD instantly without SHAP overhead.")

    with st.form("inference_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pickup = st.text_input("Pickup city", "Lexington")
            delivery = st.text_input("Delivery city", "Fort Wayne")
            distance = st.number_input("Distance (miles)", min_value=1.0, value=360.0)
            equipment = st.selectbox("Equipment type", ["Dry Van", "Reefer", "Flatbed"])
        
        with col2:
            weight = st.number_input("Weight (lbs)", min_value=100.0, value=32000.0)
            date = st.date_input("Pickup date")
            market_index = st.number_input("Market index", value=1.05)
            quote_signal = st.number_input("Quote signal", value=1.0)

        with col3:
            pickup_lat = st.number_input("Pickup latitude", value=38.0464)
            pickup_lon = st.number_input("Pickup longitude", value=-84.4970)
            delivery_lat = st.number_input("Delivery latitude", value=41.0793)
            delivery_lon = st.number_input("Delivery longitude", value=-85.1394)

        submitted = st.form_submit_button("Calculate freight rate", type="primary")

    if submitted:
        payload = {
            "load_id": "UI-TEST-001",
            "pickup": pickup,
            "delivery": delivery,
            "distance": distance,
            "equipment": equipment,
            "weight": weight,
            "date": str(date),
            "market_index": market_index,
            "quote_signal": quote_signal,
            "pickup_lat": pickup_lat,
            "pickup_lon": pickup_lon,
            "delivery_lat": delivery_lat,
            "delivery_lon": delivery_lon
        }

        with st.spinner("Running prediction..."):
            try:
                prediction_res = predict_freight_rate(payload)
                st.success("Prediction generated successfully!")
                
                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.metric("Predicted rate: ", f"${prediction_res['predicted_rate_usd']:,.2f}")
                res_col2.metric("Model engine", prediction_res['model_engine'])
                res_col3.metric("Model version", prediction_res['model_version'])

            except Exception as e:
                st.error(f"Inference failed: {str(e)}")


# for batch prediction
elif app_mode == "Batch validation pipeline":
    st.header("Batch validation & scoring pipeline")
    st.markdown("Run batch inference across the entire validation dataset and generate output predictions template files.")

    if st.button("Execute batch validation", type="primary"):
        with st.spinner("Processing batch validation records..."):
            try:
                from src.v1.prediction import run_batch_validation
                run_batch_validation()
                st.success("Batch validation completed successfully! File generated: `validation_predictions.csv`")
                
                output_csv = PROJECT_ROOT / "validation_predictions.csv"
                if output_csv.exists():
                    df_out = pd.read_csv(output_csv)
                    st.dataframe(df_out.head(20))
                    
                    with open(output_csv, "rb") as f:
                        st.download_button(
                            label="Download predictions CSV",
                            data=f,
                            file_name="validation_predictions.csv",
                            mime="text/csv"
                        )
            except Exception as e:
                st.error(f"Batch validation failed: {str(e)}")

# december chart batch prediction
elif app_mode == "December chart batch prediction":
    st.header("December chart batch prediction")

    st.markdown(
        "Generate freight-rate predictions for every record in "
        "`data/raw/december-chart-inputs.csv`. "
        "The existing file will be updated in place, with predictions "
        "written to the `predicted_rate` column."
    )

    input_csv = PROJECT_ROOT / "data/raw/december-chart-inputs.csv"

    if input_csv.exists():
        st.caption(f"Target file: {input_csv}")

        try:
            df_preview = pd.read_csv(input_csv)

            st.write(f"Records to process: **{len(df_preview):,}**")

            if "predicted_rate" in df_preview.columns:
                st.dataframe(df_preview.head(10), width="stretch")
            else:
                st.warning(
                    "The `predicted_rate` column is not present in "
                    "the input file. Check the CSV before running predictions."
                )

        except Exception as e:
            st.error(f"Could not read the chart input file: {e}")

    else:
        st.error(f"Input file not found: {input_csv}")

    if st.button(
        "Generate December chart predictions",
        type="primary",
        disabled=not input_csv.exists()
    ):
        with st.spinner(
            "Generating predictions and updating the December chart file..."
        ):
            try:
                from src.v1.prediction import run_batch_validation_dec

                result = run_batch_validation_dec()

                st.success(
                    "December chart predictions completed. "
                    "The existing CSV has been updated."
                )

                # reload the updated file to show the saved results.
                if input_csv.exists():
                    df_updated = pd.read_csv(input_csv)

                    st.subheader("Updated chart data")
                    st.dataframe(df_updated.head(20), width="stretch")

                    if "predicted_rate" in df_updated.columns:
                        st.metric(
                            "Records updated",
                            f"{df_updated['predicted_rate'].notna().sum():,}"
                        )

                    with open(input_csv, "rb") as f:
                        st.download_button(
                            label="Download updated December chart CSV",
                            data=f,
                            file_name="december-chart-inputs.csv",
                            mime="text/csv"
                        )

                if result is not None:
                    with st.expander("Execution details"):
                        st.write(result)

            except Exception as e:
                st.error(
                    f"December chart batch prediction failed: {str(e)}"
                )