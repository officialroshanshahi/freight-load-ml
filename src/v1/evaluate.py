import datetime, joblib, pandas as pd, numpy as np, matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.v1.config import PROJECT_ROOT


# for plotting the evaluation metrics
def eval_plots(best_models, y_test_usd, ver=1):
    print("\nGenerating evaluation plots...")
    rows = []
    
    # getting the metrics out of the best_models artifact
    for name, r in best_models.items():
        rows.append({
            "model": name,
            "val_r2": r["val_metrics"]["r2"],
            "val_rmse": r["val_metrics"]["rmse"],
            **r["test_metrics"]
        })

    df = pd.DataFrame(rows)
    
    # for saving those evaluation plots
    eval_dir = PROJECT_ROOT / f"plots/evaluate/v{ver}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # for giving unique name to every single metrics plots
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    df.to_csv(eval_dir / f"model_metrics_v{ver}_{timestamp}.csv", index=False)
    df.to_excel(eval_dir / f"model_metrics_v{ver}_{timestamp}.xlsx", index=False)

    generated_images = []

    # saving the plots and converting the fig for frontend
    def save_fig(filename):
        disk_path = eval_dir / f"{filename}_v{ver}_{timestamp}.png"
        plt.savefig(disk_path, format='png', dpi=150, bbox_inches='tight')
        plt.close()

    # plotting the required metrics
    fig, ax = plt.subplots(figsize=(10, 5))
    
    df.set_index("model")[["val_r2", "r2"]].plot(
        kind="bar", ax=ax, figsize=(10, 5), color=['#4361ee', '#4cc9f0']
    )
    
    plt.title("Model accuracy baseline performance comparison (Log-Scale R² scores)")
    plt.ylabel("Score value")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=.3)
    plt.tight_layout()
    
    generated_images.append(save_fig("performance_comparison"))

    # plotting continuous actual vs predicted regression fit scatters
    for name, r in best_models.items():
        plt.figure(figsize=(9, 6))
        
        preds_usd = r["predictions_usd"]
        
        plt.scatter(y_test_usd, preds_usd, alpha=0.4, color='#3a0ca3', edgecolor='k', s=20)
        
        ideal_line = [y_test_usd.min(), y_test_usd.max()]
        
        plt.plot(ideal_line, ideal_line, "r--", linewidth=2, label="Perfect predictions alignment line")
        
        plt.xlabel("Actual freight rate (USD)")
        plt.ylabel("Predicted freight rate (USD)")
        plt.title(f"Continuous regression fit analysis scatter: {name}")
        plt.legend(loc='upper left')
        plt.grid(alpha=.3)
        plt.tight_layout()
        
        generated_images.append(save_fig(f"{name}_regression_fit"))

    print(f"Metrics and plots successfully saved to: {eval_dir}/")
    print(df.round(4))
    
    return generated_images


# main evaluation logic
def evaluator(x_test_processed, x_test_processed_std, y_test, trained_models, plot='no', ver=1):
    
    best_models = {}
    val_scores = {}
    
    val_scores_path = PROJECT_ROOT / f"models/ev_search/v{ver}/val_scores_{ver}.joblib"
    
    if val_scores_path.exists():
        val_scores = joblib.load(val_scores_path)

    print("\nEvaluating models on test data continuous parameters spectrum...")

    # keep log scale reference for R² calculation
    y_test_log = y_test.squeeze()
    
    # convert y back to USD for monetary error evaluations (RMSE, MAE)
    y_test_usd = np.expm1(y_test_log)

    for model_name, model in trained_models.items():
        m_lower = model_name.lower()
        
        if any(k in m_lower for k in ["logisticregression", "linearregression", "ridge", "lasso", "kneighbor", "svr"]):
            X_te = x_test_processed_std
        else:
            X_te = x_test_processed

        # calculate prediction outputs (returns log-scale values)
        y_probs_log = model.predict(X_te)
        
        # map continuous arrays back to absolute dollar formats safely for USD metrics
        y_preds_usd = np.expm1(y_probs_log)
        y_preds_usd = np.clip(y_preds_usd, a_min=0.0, a_max=None)

        # compute metrics: R² on log scale (objective match), RMSE/MAE in actual USD
        test_r2 = r2_score(y_test_log, y_probs_log)
        test_rmse = np.sqrt(mean_squared_error(y_test_usd, y_preds_usd))
        test_mae = mean_absolute_error(y_test_usd, y_preds_usd)

        print(f"[{model_name}] Test Results -> R² (log): {test_r2:.4f} | RMSE: ${test_rmse:.2f} | MAE: ${test_mae:.2f}")

        model_val_metrics = val_scores.get(model_name, {"r2": 0.0, "rmse": np.inf})

        best_models[model_name] = {
            'model': model,
            'val_metrics': model_val_metrics,
            'predictions_usd': y_preds_usd,
            'test_metrics': {
                'r2': test_r2,
                'rmse': test_rmse,
                'mae': test_mae
            }
        }
    
    # dumping the model artifacts
    best_models_path = PROJECT_ROOT / f'models/ev_metrics/v{ver}/best_models_metrics_{ver}.joblib'
    best_models_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_models, best_models_path)

    best_model_name, _ = max(
        best_models.items(), 
        key=lambda item: item[1]['test_metrics']['r2']
    )

    # model metrics
    best_model = best_models[best_model_name]['model']
    print(f"\nBest model selected: {best_model_name} with Test R² (log): {best_models[best_model_name]['test_metrics']['r2']:.4f}")
    
    best_model_path = PROJECT_ROOT / f"models/models/v{ver}/best_model_overall_{ver}.pkl"
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, best_model_path)

    generated_plots = []
    
    if str(plot).lower() == "yes":
        generated_plots = eval_plots(best_models, y_test_usd, ver=ver)
    
    return best_model_name, best_models[best_model_name], generated_plots