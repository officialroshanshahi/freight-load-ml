import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from src.v1.config import PIPELINE_PATH, RAW_DATA_PATH, SCALER_PATH

# creating a custom transformer for feature engineering
class CustomFreightFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self): 
        self.weight_median_map_ = {}
        self.global_weight_median_ = 31486.5
        self.global_market_median_ = 1.0558
        self.is_fitted_ = False
        
    def fit(self, X, y=None):
        x_copy = pd.DataFrame(X).copy()
        
        # for handling negative weight and imputing empty values
        if 'weight' in x_copy.columns:
            x_copy['weight'] = x_copy['weight'].abs()
            
            if 'equipment' in x_copy.columns:
                self.weight_median_map_ = x_copy.groupby('equipment')['weight'].median().to_dict()
            
            self.global_weight_median_ = x_copy['weight'].median()
        
        # for imputing market_index values
        if 'market_index' in x_copy.columns:
            self.global_market_median_ = x_copy['market_index'].median()
            
        self.is_fitted_ = True
        
        return self
    
    def transform(self, X):
        x_copy = pd.DataFrame(X).copy()
        
        x_copy['weight'] = x_copy['weight'].abs()
        x_copy['weight'] = x_copy['weight'].fillna(x_copy['equipment'].map(self.weight_median_map_))
        x_copy['weight'] = x_copy['weight'].fillna(self.global_weight_median_)
        x_copy['market_index'] = x_copy['market_index'].fillna(self.global_market_median_)
        
        # for haversine distance and directional bearings
        lat1, lon1, lat2, lon2 = map(
            np.radians, [x_copy['pickup_lat'], x_copy['pickup_lon'], x_copy['delivery_lat'], x_copy['delivery_lon']]
        )
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        x_copy['haversine_distance'] = 2 * 3956 * np.arcsin(np.sqrt(a))
        
        x_copy['route_circuity'] = x_copy['distance'] / (x_copy['haversine_distance'] + 1e-5)
    
        x_copy['log_distance'] = np.log1p(x_copy['distance'])
        
        # heading angle from pickup to delivery
        bx = np.sin(dlon) * np.cos(lat2)
        by = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
        x_copy['pickup_delivery_bearing'] = np.arctan2(bx, by)
        
        # midpoints of lat and lon
        x_copy['midpoint_lat'] = (x_copy['pickup_lat'] + x_copy['delivery_lat']) / 2
        x_copy['midpoint_lon'] = (x_copy['pickup_lon'] + x_copy['delivery_lon']) / 2
        
        # base and advanced interaction features
        x_copy['weight_per_mile'] = x_copy['weight'] / x_copy['distance']
        x_copy['market_tension_index'] = x_copy['market_index'] * x_copy['quote_signal']
        x_copy['total_payload_weight'] = x_copy['weight_per_mile'] * x_copy['distance']
        x_copy['excess_distance'] = x_copy['distance'] * (x_copy['route_circuity'] - 1)
        
        # creating bins for the distance: haul type
        bins = [0, 250, 750, 1500, float('inf')]
        labels = ['Short_Haul', 'Regional', 'Medium_Haul', 'Long_Haul']
        x_copy['haul_type'] = pd.cut(x_copy['distance'], bins=bins, labels=labels).astype(str)
        
        # dropping unnecessary features
        drop_cols = ['load_id', 'pickup', 'delivery', 'date', 'weight', 'market_index']
        x_copy.drop(columns=[c for c in drop_cols if c in x_copy.columns], inplace=True)
        
        return x_copy
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None: return None
        
        dropped = ['load_id', 'pickup', 'delivery', 'date', 'weight', 'market_index']
        base = [f for f in input_features if f not in dropped]
        
        new_feats = [
            'haversine_distance', 'route_circuity', 'log_distance', 'pickup_delivery_bearing',
            'midpoint_lat', 'midpoint_lon', 'weight_per_mile', 'market_tension_index', 
            'total_payload_weight', 'excess_distance', 'haul_type'
        ]
        
        return np.array(base + new_feats)

# the main logic behind the preprocessing pipeline,
# which includes imputation, feature engineering, and scaling
def preprocessor(data, fit_transform='yes', std='yes', ver=1):
    data_copy = data.copy()
    
    pipeline_path, scaler_path = PIPELINE_PATH, SCALER_PATH

    # for fit_transform == yes i.e. training set
    if fit_transform.lower().strip() == 'yes':
        impute_full = ColumnTransformer(transformers=[
            ('cat_', SimpleImputer(strategy='most_frequent'), make_column_selector(dtype_exclude='number', dtype_include=['category', 'string'])),
            ('num_', SimpleImputer(strategy='median'), make_column_selector(dtype_exclude=['category', 'string'], dtype_include='number'))
        ], verbose_feature_names_out=False, remainder='passthrough')
        
        categorical_cols = ['equipment', 'haul_type']
        encoder_transformer = ColumnTransformer(transformers=[
            ('cat_encode', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
        ], remainder='passthrough', verbose_feature_names_out=False)
        
        pipeline_normal = Pipeline(steps=[
            ('imputer', impute_full),
            ('feature_eng', CustomFreightFeatureEngineer()),
            ('encoding', encoder_transformer)
        ])
        
        pipeline_normal.set_output(transform="pandas")
        
        data_transformed = pipeline_normal.fit_transform(data_copy)
        
        pipeline_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline_normal, pipeline_path)
        
        print('Data preprocessing complete.\n')
        
        if std.lower().strip() == 'yes':
            data_transformed_std = data_transformed.copy()
            
            continuous_cols = [
                col for col in data_transformed.columns 
                if col not in ['posted_rate', 'log_posted_rate'] 
                and not col.startswith('equipment') 
                and not col.startswith('haul_type')
            ]
            
            scaler = RobustScaler()
            scaler.set_output(transform="pandas")
            
            data_transformed_std[continuous_cols] = scaler.fit_transform(data_transformed_std[continuous_cols])
            
            scaler_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, scaler_path)
            
            print('Returning the processed dataset with RobustScaler.\n')
            return data_transformed, data_transformed_std
        
        print('Returning the processed dataset without scaling.\n')
        return data_transformed
    
    # for fit_transform == no i.e. test or prediction set
    else:
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline file not found at {pipeline_path}.")
        
        pipeline_normal = joblib.load(pipeline_path)
        pipeline_normal.set_output(transform="pandas")
        
        data_transformed = pipeline_normal.transform(data_copy)
        
        print('Data preprocessing complete.\n')
        
        if std.lower().strip() == 'yes':
            if not scaler_path.exists():
                raise FileNotFoundError(f"Scaler file not found at {scaler_path}.")
            
            data_transformed_std = data_transformed.copy()
            continuous_cols = [
                col for col in data_transformed.columns 
                if col not in ['posted_rate', 'log_posted_rate'] 
                and not col.startswith('equipment_') 
                and not col.startswith('haul_type_')
            ]
            
            scaler = joblib.load(scaler_path)
            scaler.set_output(transform="pandas")
            
            data_transformed_std[continuous_cols] = scaler.transform(data_transformed_std[continuous_cols])
            
            print('Returning the processed dataset with scaling.\n')
            return data_transformed, data_transformed_std
        
        print('Returning the processed dataset without scaling.\n')
        return data_transformed


# y preprocessor
def y_preprocessor(y, ver=1):
    y_copy = y.copy()
    y_log = np.log1p(y_copy.astype('float64'))
    return y_log

# for loading the raw data and splitting it
def load_data(ver=1):
    raw_data_path = RAW_DATA_PATH
    data = pd.read_csv(raw_data_path)
    data = data.drop_duplicates()
    
    X = data.drop(columns='posted_rate')
    y = data['posted_rate'].copy()
    
    y = y_preprocessor(y, ver=ver) 
    
    if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
        X = X.loc[y.index]
    
    y_binned = pd.qcut(y, q=6, labels=False, duplicates="drop")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y_binned
    )
    
    print("Data loading complete. Moving on to the next step in the pipeline.\n")
    
    return X_train, X_test, y_train, y_test