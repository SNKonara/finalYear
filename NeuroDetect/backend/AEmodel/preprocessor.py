import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Data preprocessing for fraud detection"""
    
    def __init__(self, scaler_path=None):
        """
        Initialize preprocessor
        
        Args:
            scaler_path: Path to save/load scaler
        """
        self.scaler = StandardScaler()
        self.scaler_path = scaler_path
        self.feature_columns = None
        self.top_categories = None  # Store top categories from training
        self.category_columns = None  # Store category dummy column names
        
    def preprocess(self, df, is_training=True, save_scaler=False):
        """
        Preprocess raw transaction data
        
        Args:
            df: Raw DataFrame with required columns
            is_training: Whether this is training data
            save_scaler: Whether to save the scaler
            
        Returns:
            Preprocessed numpy array, labels, feature columns
        """
        # Keep only essential columns (adjust based on your dataset)
        columnsN = [
            'amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long',
            'is_fraud', 'trans_date_trans_time', 'category', 'gender'
        ]
        
        # Check if all required columns exist
        missing_cols = [col for col in columnsN if col not in df.columns]
        if missing_cols:
            # Try without is_fraud for inference
            columnsN_inference = [col for col in columnsN if col != 'is_fraud']
            missing_cols_inference = [col for col in columnsN_inference if col not in df.columns]
            if missing_cols_inference:
                raise ValueError(f"Missing required columns: {missing_cols_inference}")
            df = df[columnsN_inference].copy()
        else:
            df = df[columnsN].copy()
        
        # Convert timestamp
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        
        # Time features
        df['hour'] = df['trans_date_trans_time'].dt.hour
        df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
        df['day_of_month'] = df['trans_date_trans_time'].dt.day
        df['month'] = df['trans_date_trans_time'].dt.month
        
        # Spatial features
        df['distance'] = np.sqrt(
            (df['lat'] - df['merch_lat'])**2 +
            (df['long'] - df['merch_long'])**2
        )
        
        # Transaction amount features
        df['log_amt'] = np.log1p(df['amt'])
        df['amt_per_pop'] = df['amt'] / (df['city_pop'] + 1)
        
        # Time-based features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Categorical encoding
        if is_training:
            # During training: identify top categories
            self.top_categories = df['category'].value_counts().nlargest(8).index.tolist()
            df['category_top'] = df['category'].apply(
                lambda x: x if x in self.top_categories else 'other'
            )
            category_dummies = pd.get_dummies(df['category_top'], prefix='cat', drop_first=True)
            
            # Store the category column names for inference
            self.category_columns = category_dummies.columns.tolist()
            
        else:
            # During inference: use stored top categories
            if self.top_categories is None:
                raise ValueError("Preprocessor must be trained first or top_categories must be set")
            
            df['category_top'] = df['category'].apply(
                lambda x: x if x in self.top_categories else 'other'
            )
            category_dummies = pd.get_dummies(df['category_top'], prefix='cat', drop_first=True)
            
            # Ensure all training columns exist
            for col in self.category_columns:
                if col not in category_dummies.columns:
                    category_dummies[col] = 0
            
            # Keep only training columns in the same order
            category_dummies = category_dummies[self.category_columns]
        
        df = pd.concat([df, category_dummies], axis=1)
        
        # Gender encoding
        df['gender_M'] = (df['gender'] == 'M').astype(int)
        
        # Drop original columns
        df = df.drop(['trans_date_trans_time', 'category', 'gender', 'category_top'], axis=1)
        
        # Store feature columns (only during training)
        if is_training:
            self.feature_columns = [col for col in df.columns if col != 'is_fraud']
        
        # Separate features and target
        if 'is_fraud' in df.columns:
            X = df.drop('is_fraud', axis=1).values
            y = df['is_fraud'].values
        else:
            X = df.values
            y = None
        
        # Ensure correct number of features during inference
        if not is_training and self.feature_columns is not None:
            if X.shape[1] != len(self.feature_columns):
                raise ValueError(
                    f"Feature mismatch: expected {len(self.feature_columns)} features, "
                    f"got {X.shape[1]} features"
                )
        
        # Scale features
        if is_training:
            X_scaled = self.scaler.fit_transform(X)
            if save_scaler and self.scaler_path:
                self.save_preprocessor(self.scaler_path)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled, y, self.feature_columns
    
    def save_preprocessor(self, base_path):
        """Save scaler and preprocessing parameters"""
        # Save scaler
        joblib.dump(self.scaler, base_path)
        
        # Save preprocessing parameters
        params_path = base_path.replace('.pkl', '_params.pkl')
        params = {
            'feature_columns': self.feature_columns,
            'top_categories': self.top_categories,
            'category_columns': self.category_columns
        }
        joblib.dump(params, params_path)
        print(f"✅ Saved preprocessor to {base_path}")
        print(f"✅ Saved parameters to {params_path}")
    
    def load_preprocessor(self, base_path):
        """Load scaler and preprocessing parameters"""
        # Load scaler
        self.scaler = joblib.load(base_path)
        
        # Load preprocessing parameters
        params_path = base_path.replace('.pkl', '_params.pkl')
        params = joblib.load(params_path)
        
        self.feature_columns = params['feature_columns']
        self.top_categories = params['top_categories']
        self.category_columns = params['category_columns']
        
        print(f"✅ Loaded preprocessor from {base_path}")
        print(f"   Features: {len(self.feature_columns)}")
        print(f"   Top categories: {self.top_categories}")
        print(f"   Category columns: {self.category_columns}")
        
        return self
    
    def load_scaler(self, scaler_path):
        """Load pre-trained scaler (backward compatibility)"""
        return self.load_preprocessor(scaler_path)