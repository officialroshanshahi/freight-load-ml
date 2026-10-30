import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from src.v1.config import PROJECT_ROOT


# wrapper for the Advanced Weighted Ensemble
class AdvancedEnsembleModel:
    def __init__(self, models_dict, weights):
        self.models_dict = models_dict
        self.weights = weights
        
    def predict(self, X):
        pred_log = 0.0
        for name, model in self.models_dict.items():
            pred_log += self.weights.get(name, 0.0) * model.predict(X)
        return pred_log

    @property
    def feature_names_in_(self):
        for model in self.models_dict.values():
            if hasattr(model, "feature_names_in_"):
                return model.feature_names_in_
        return None


# main training pipeline
def trainer(x_train, x_train_pr, y_train, model_dic, params, scoring_metrics, ver):
    
    print('\nStarting training pipeline with hyperparameter optimization.\n')
    
    best_models = {}
    val_scores = {}
    val_preds_log_dict = {}
    
    y_train_flat = y_train.squeeze()
    
    # discretize target for balanced stratification during train/val split
    y_binned = pd.qcut(y_train_flat, q=6, labels=False, duplicates='drop')
    train, val = train_test_split(np.arange(len(x_train)), test_size=0.20, stratify=y_binned, random_state=42)
    
    split_fold = np.zeros(len(x_train))
    split_fold[train] = -1
    pds = PredefinedSplit(test_fold=split_fold)
    
    val_y_log = y_train_flat.iloc[val]
    val_y_usd = np.expm1(val_y_log)
    
    for m_name, init_model in model_dic.items():
        
        print(f"Searching space for {m_name}...")
        m_low = m_name.lower()
        
        # select scaled dataset for linear models, raw/processed for tree models
        X_tr = x_train_pr.copy() if any(k in m_low for k in ["logisticregression", "linearregression", "ridge", "lasso", "kneighbor", "svr"]) else x_train.copy()
        
        n_workers = -1
        fit_params = {}
        
        if "xgbregressor" in m_low or "lgbmregressor" in m_low:
            init_model.set_params(n_estimators=1000, learning_rate=0.05, random_state=42)
            if "xgbregressor" in m_low:
                init_model.set_params(tree_method="hist", eval_metric="rmse", early_stopping_rounds=15)
                fit_params = {"eval_set": [(X_tr.iloc[val], y_train_flat.iloc[val])], "verbose": False}

        search = RandomizedSearchCV(
            estimator=init_model,
            param_distributions=params[m_name],
            n_iter=10,
            cv=pds,
            scoring=scoring_metrics if isinstance(scoring_metrics, str) else scoring_metrics.get(m_name, 'r2'),
            refit=True,
            random_state=42,
            n_jobs=n_workers
        )
        
        search.fit(X_tr, y_train_flat, **fit_params)
        best_model = search.best_estimator_

        # store validation log predictions for ensemble construction
        val_preds_log = best_model.predict(X_tr.iloc[val])
        val_preds_log_dict[m_name] = val_preds_log
        
        val_preds_usd = np.clip(np.expm1(val_preds_log), a_min=0.0, a_max=None)
        val_rmse = np.sqrt(mean_squared_error(val_y_usd, val_preds_usd))
        val_r2 = r2_score(val_y_log, val_preds_log)
        
        print(f"[{m_name}] Validation R² score (log scale): {val_r2:.4f} | Validation RMSE: ${val_rmse:.2f}")

        # save individual model artifacts
        artifacts = [
            (f"models/models/v{ver}/{m_name}_best_{ver}.pkl", best_model),
            (f"models/ev_search/v{ver}/{m_name}_search_{ver}.pkl", search)
        ]
        
        for path, obj in artifacts:
            (PROJECT_ROOT / path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(obj, PROJECT_ROOT / path)

        best_models[m_name] = best_model
        val_scores[m_name] = {"r2": val_r2, "rmse": val_rmse}
        
        print(f"Search and validation parameter tracking complete for {m_name}.\n")

    print("Building and evaluating Advanced Weighted Ensemble model...")
    
    # dynamic key matching to ensure mapping regardless of dictionary name casing
    def find_key(mapping, keywords):
        for k in mapping.keys():
            if all(kw in k.lower() for kw in keywords):
                return k
        return None

    lgb_key = find_key(best_models, ['lgb']) or 'LightGBM'
    xgb_key = find_key(best_models, ['xgb']) or 'XGBoost'
    rf_key = find_key(best_models, ['forest']) or 'RandomForest'

    # this can be changed as necessary (or can be tuned as hyperparameters)
    ensemble_weights = {
        lgb_key: 0.2, 
        xgb_key: 0.4, 
        rf_key: 0.4
    }
    
    # compute weighted validation predictions
    ensemble_val_preds_log = sum(
        weight * val_preds_log_dict[k] 
        for k, weight in ensemble_weights.items()
        if k in val_preds_log_dict
    )
    
    ensemble_val_preds_usd = np.clip(np.expm1(ensemble_val_preds_log), a_min=0.0, a_max=None)
    
    ens_val_rmse = np.sqrt(mean_squared_error(val_y_usd, ensemble_val_preds_usd))
    ens_val_r2 = r2_score(val_y_log, ensemble_val_preds_log)
    
    print(f"[AdvancedEnsemble] Validation R² score (log scale): {ens_val_r2:.4f} | Validation RMSE: ${ens_val_rmse:.2f}")
    
    advanced_ensemble_model = AdvancedEnsembleModel(
        models_dict={
            lgb_key: best_models[lgb_key], 
            xgb_key: best_models[xgb_key], 
            rf_key: best_models[rf_key]
        },
        weights=ensemble_weights
    )
    
    # save the advanced ensemble artifact
    ensemble_path = PROJECT_ROOT / f"models/models/v{ver}/AdvancedEnsemble_best_{ver}.pkl"
    ensemble_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(advanced_ensemble_model, ensemble_path)
    
    best_models['AdvancedEnsemble'] = advanced_ensemble_model
    val_scores['AdvancedEnsemble'] = {"r2": ens_val_r2, "rmse": ens_val_rmse}
    
    print("Advanced ensemble successfully created and registered.\n")

    print("All models have been trained and configurations optimized.")
    
    # save validation metrics tracker
    (PROJECT_ROOT / f"models/ev_search/v{ver}/val_scores_{ver}.joblib").parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(val_scores, PROJECT_ROOT / f"models/ev_search/v{ver}/val_scores_{ver}.joblib")
    
    return best_models, val_scores