import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os

class DataPreprocessor:
    """Data preprocessing for NVMe failure prediction"""
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load CSV data"""
        print(f"Loading data from {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {self.df.columns.tolist()}")
        return self.df
    
    def explore_data(self):
        """Basic data exploration"""
        print("\n=== DATA EXPLORATION ===")
        print(f"\nMissing values:\n{self.df.isnull().sum()}")
        print(f"\nData types:\n{self.df.dtypes}")
        print(f"\nTarget distribution:\n{self.df['Failure_Flag'].value_counts()}")
        print(f"\nBasic statistics:\n{self.df.describe()}")
        
    def handle_missing_values(self):
        """Handle missing values"""
        print("\nHandling missing values...")
        # Fill missing values with median for numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        print("Missing values handled")
        
    def encode_categorical(self):
        """Encode categorical variables"""
        print("\nEncoding categorical variables...")
        categorical_cols = ['Vendor', 'Model', 'Firmware_Version']
        
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
            print(f"  {col}: {len(le.classes_)} unique values")
        
    def create_features(self):
        """Create additional features"""
        print("\nCreating new features...")
        
        # Interaction features
        self.df['Power_Temp_Ratio'] = self.df['Power_On_Hours'] / (self.df['Temperature_C'] + 1)
        self.df['Error_Sum'] = (self.df['Media_Errors'] + 
                               self.df['Unsafe_Shutdowns'] + 
                               self.df['CRC_Errors'])
        self.df['Error_Rate_Sum'] = self.df['Read_Error_Rate'] + self.df['Write_Error_Rate']
        self.df['Wear_Temp_Ratio'] = self.df['Percent_Life_Used'] / (self.df['Temperature_C'] + 1)
        
        print(f"  Created 4 new features")
        
    def remove_features(self):
        """Remove non-predictive features"""
        print("\nRemoving non-predictive features...")
        features_to_drop = ['Drive_ID', 'Failure_Mode']  # ID and target-related
        self.df.drop(features_to_drop, axis=1, inplace=True)
        print(f"  Dropped: {features_to_drop}")
        
    def prepare_train_test(self, test_size=0.2, random_state=42):
        """Prepare training and test sets"""
        print("\nPreparing train/test split...")
        
        # Separate features and target
        X = self.df.drop('Failure_Flag', axis=1)
        y = self.df['Failure_Flag']
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y, shuffle=True
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"  Training set: {X_train_scaled.shape}")
        print(f"  Test set: {X_test_scaled.shape}")
        print(f"  Target distribution (train): {np.bincount(y_train)}")
        print(f"  Target distribution (test): {np.bincount(y_test)}")
        
        return (X_train_scaled, X_test_scaled, y_train, y_test, 
                X_train.columns.tolist())
    
    def save_preprocessed_data(self, output_dir, X_train, X_test, y_train, y_test):
        """Save preprocessed data and encoders"""
        print(f"\nSaving preprocessed data to {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
        
        # Save encoders and scaler
        import pickle
        with open(os.path.join(output_dir, 'encoders.pkl'), 'wb') as f:
            pickle.dump(self.label_encoders, f)
        with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print("Data saved successfully")
        
    def preprocess(self, output_dir='./processed_data'):
        """Run complete preprocessing pipeline"""
        self.load_data()
        self.explore_data()
        self.handle_missing_values()
        self.encode_categorical()
        self.create_features()
        self.remove_features()
        
        X_train, X_test, y_train, y_test, feature_names = self.prepare_train_test()
        self.save_preprocessed_data(output_dir, X_train, X_test, y_train, y_test)
        
        return X_train, X_test, y_train, y_test, feature_names


if __name__ == "__main__":
    # Setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, '../data/NVMe_Drive_Failure_Dataset.csv')
    output_dir = os.path.join(current_dir, 'processed_data')
    
    # Preprocess
    preprocessor = DataPreprocessor(csv_path)
    X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess(output_dir)
    
    print("\n=== PREPROCESSING COMPLETE ===")
    print(f"Feature count: {len(feature_names)}")
    print(f"Features: {feature_names}")
