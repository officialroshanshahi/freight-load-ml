import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

# figuring out the project root directory dynamically, regardless of where the script is run from
PROJECT_ROOT = Path(__file__).resolve().parents[2] if '__file__' in locals() else Path.cwd()

# global version marker to keep track of iterations
VERSION = 1
ACTIVE = PROJECT_ROOT / f"models/models/v{VERSION}/active_model.json"

# target naming vectors and directory configs
TARGET_COL = 'posted_rate'
TRANSFORMED_TARGET_COL = 'log_posted_rate'

# project structural path mappings
RAW_DATA_PATH = PROJECT_ROOT / "data/raw/train-test.csv"
PIPELINE_PATH = PROJECT_ROOT / f'models/pipelines/v{VERSION}/pipeline_normal_v{VERSION}.joblib'
SCALER_PATH = PROJECT_ROOT / f'models/scalers/v{VERSION}/scaler_v{VERSION}.joblib'

# model configs:
MODELS = {
    'LightGBM': lgb.LGBMRegressor(random_state=42, n_jobs=-1, force_col_wise=True, verbose=-1),
    # using squarederror as loss function as it is standard objective
    'XGBoost': xgb.XGBRegressor(random_state=42, objective='reg:squarederror', n_jobs=-1),
    'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1)
}

PARAMS = {
    'LightGBM': {
        'n_estimators': [500, 800, 1000], # number of trees built so, keeping it low to high (pushed higher since 500 won)
        'learning_rate': [0.01, 0.03, 0.05], # focused around 0.03
        'max_depth': [3, 4, 5], # checking deep tree or small one (centered around 4)
        'num_leaves': [15, 31, 45], # width of the model 
        'subsample': [0.7, 0.8, 0.9], # for reducing overfitting, using low to high
        'colsample_bytree': [0.8, 0.9, 1.0], # fraction of sample looked for best spliting
        'reg_alpha': [0.0, 0.01], # for sparsity
        'reg_lambda': [0.1, 0.5, 1.0], # for smoothing the weight (shifted slightly higher since 0.5 won)
        'min_child_weight': [3, 5, 7] # amount of data required to split
    },
    'XGBoost': {
        'n_estimators': [1000, 1200, 1500], # number of trees built so, keeping it low to high (pushed higher since 1000 hit the ceiling)
        'learning_rate': [0.01, 0.02, 0.03], # shifted lower to pair with 1000+ trees
        'max_depth': [2, 3, 4], # checking deep tree or small one (centered around 3)
        'subsample': [0.7, 0.8, 0.9], # for reducing overfitting, using low to high
        'colsample_bytree': [0.8, 0.9, 1.0], # fraction of sample looked for best spliting
        'min_child_weight': [10, 15, 20], # amount of data required to split, keeping it low to high (pushed higher since 10 hit the ceiling)
        'reg_alpha': [0.0, 0.01, 0.1], # for sparsity
        'reg_lambda': [50.0, 75.0, 100.0] # for smoothing the weight (pushed higher since 50.0 hit the ceiling)
    },
    'RandomForest': {
        'n_estimators': [300, 400, 500], # number of trees built so, keeping it low to high (focused around 400)
        'max_depth': [15, 20, 25], # checking deep tree or small one
        'min_samples_split': [15, 20, 25], # checking for high or low sample limit to split the node (shifted higher since 15 won)
        'min_samples_leaf': [10, 12, 15], # amount of data required to split, keeping it low to high (centered around 12)
        'max_features': [0.4, 0.5, 0.6] # maximum number to be looked when checking for best split (shifted lower since 0.6 won)
    }
}