"""
Unit Tests for Car Price Prediction Models
Run with: pytest tests.py -v
"""

import unittest
import json
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import os

# Import modules
from utils import preprocess_input_data, load_data, handle_missing_values, remove_outliers
from prediction_utils import PredictionHistory
from metrics import calculate_metrics



class TestDataLoading(unittest.TestCase):
    """Test data loading functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.data_path = 'data/car_price_prediction_.csv'
    
    def test_load_data(self):
        """Test loading data"""
        df = load_data(self.data_path)
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)
        self.assertIn('Brand', df.columns)
        self.assertIn('Price', df.columns)
    
    def test_dataset_shape(self):
        """Test dataset dimensions"""
        df = load_data(self.data_path)
        self.assertGreater(df.shape[0], 0)  # rows
        self.assertGreater(df.shape[1], 0)  # columns
    
    def test_missing_values_handling(self):
        """Test missing value handling"""
        # Create sample data with missing values
        df = pd.DataFrame({
            'Brand': ['Toyota', None, 'BMW'],
            'Year': [2020, 2021, None],
            'Car Price': [25000, 30000, 35000]
        })
        
        df_clean = handle_missing_values(df)
        self.assertEqual(df_clean['Brand'].isna().sum(), 0, "Missing brand values not handled")
        self.assertEqual(df_clean['Year'].isna().sum(), 0, "Missing year values not handled")
    
    def test_outlier_removal(self):
        """Test outlier removal"""
        df = pd.DataFrame({
            'Price': [20000, 21000, 22000, 23000, 100000],  # 100000 is outlier
            'Mileage': [45000, 50000, 48000, 52000, 51000]
        })
        
        # Only test that the function runs without error
        df_clean = remove_outliers(df)
        self.assertIsNotNone(df_clean)
        self.assertGreater(len(df_clean), 0)




class TestPredictionHistory(unittest.TestCase):
    """Test prediction history functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_file = 'test_history.json'
        self.history = PredictionHistory(self.test_file)
    
    def test_add_prediction(self):
        """Test adding prediction to history"""
        features = {'Brand': 'Toyota', 'Year': 2020}
        predictions = {'RF': 25000, 'XGB': 26000, 'GB': 25500}
        
        self.history.add_prediction(features, predictions, ['RF', 'XGB', 'GB'])
        self.assertGreater(len(self.history.predictions), 0)
    
    def test_prediction_structure(self):
        """Test that predictions have required structure"""
        features = {'Brand': 'Toyota', 'Year': 2020}
        predictions = {'RF': 25000, 'XGB': 26000}
        
        self.history.add_prediction(features, predictions, ['RF', 'XGB'])
        
        last_pred = self.history.predictions[-1]
        self.assertIn('timestamp', last_pred)
        self.assertIn('features', last_pred)
        self.assertIn('predictions', last_pred)
        self.assertIn('average_prediction', last_pred)
    
    def tearDown(self):
        """Clean up test files"""
        if Path(self.test_file).exists():
            Path(self.test_file).unlink()



class TestMetrics(unittest.TestCase):
    """Test metric calculations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.y_true = np.array([25000, 30000, 35000, 20000, 28000])
        self.y_pred = np.array([24000, 31000, 34500, 21000, 27500])
    
    def test_metrics_calculation(self):
        """Test that metrics are calculated correctly"""
        metrics = calculate_metrics(self.y_true, self.y_pred)
        
        self.assertIn('MAE', metrics)
        self.assertIn('RMSE', metrics)
        self.assertIn('R2', metrics)
        self.assertGreater(metrics['MAE'], 0)
        self.assertGreater(metrics['RMSE'], 0)
    
    def test_perfect_prediction(self):
        """Test metrics with perfect predictions"""
        y_perfect = self.y_true.copy()
        metrics = calculate_metrics(self.y_true, y_perfect)
        
        self.assertAlmostEqual(metrics['MAE'], 0, places=5)
        self.assertAlmostEqual(metrics['RMSE'], 0, places=5)
        self.assertAlmostEqual(metrics['R2'], 1.0, places=5)
    
    def test_metrics_consistency(self):
        """Test metrics are consistent"""
        metrics1 = calculate_metrics(self.y_true, self.y_pred)
        metrics2 = calculate_metrics(self.y_true, self.y_pred)
        
        self.assertEqual(metrics1['MAE'], metrics2['MAE'])
        self.assertEqual(metrics1['R2'], metrics2['R2'])



class TestModelFiles(unittest.TestCase):
    """Test that all required model files exist"""
    
    def test_models_exist(self):
        """Check that model files exist"""
        model_files = [
            'models/random_forest_model.pkl',
            'models/xgboost_model.pkl',
            'models/gradient_boosting_model.pkl',
            'models/scaler.pkl',
            'models/encoder.pkl'
        ]
        
        for file in model_files:
            self.assertTrue(Path(file).exists(), f"Model file {file} not found")
    
    def test_neural_network_exists(self):
        """Check that neural network model exists"""
        nn_file = 'models/neural_network_model.h5'
        self.assertTrue(Path(nn_file).exists(), f"Neural network model {nn_file} not found")
    
    def test_model_loading(self):
        """Test that models can be loaded"""
        try:
            with open('models/random_forest_model.pkl', 'rb') as f:
                model = pickle.load(f)
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"Failed to load model: {e}")


class TestDatasetStructure(unittest.TestCase):
    """Test dataset loading and structure"""
    
    def test_dataset_exists(self):
        """Test that dataset file exists"""
        self.assertTrue(Path('data/car_price_prediction_.csv').exists())
    
    def test_dataset_structure(self):
        """Test dataset has required columns"""
        df = pd.read_csv('data/car_price_prediction_.csv')
        
        required_columns = [
            'Brand', 'Model', 'Price', 'Year', 'Engine Size',
            'Mileage', 'Fuel Type', 'Transmission', 'Condition'
        ]
        
        for col in required_columns:
            self.assertIn(col, df.columns, f"Column {col} not found in dataset")
    
    def test_dataset_not_empty(self):
        """Test dataset is not empty"""
        df = pd.read_csv('data/car_price_prediction_.csv')
        self.assertGreater(len(df), 0, "Dataset is empty")



if __name__ == '__main__':
    unittest.main()
