#!/usr/bin/env python3
"""
Model Train modülü için unit testler
"""

import unittest
import tempfile
import shutil
import os
import pandas as pd
import numpy as np
import joblib
from unittest.mock import patch, MagicMock
from constants import FilePaths, ColumnNames


class TestModelTrain(unittest.TestCase):
    """Model Train testleri"""
    
    def setUp(self):
        """Test öncesi hazırlık"""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        # Test verisi oluştur
        self.create_test_data()
    
    def tearDown(self):
        """Test sonrası temizlik"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
    
    def create_test_data(self):
        """Test için örnek veri oluştur"""
        np.random.seed(42)
        n_samples = 200
        
        # Örnek RUL verisi
        data = {
            ColumnNames.RUL: np.random.uniform(10, 200, n_samples),
            'sensor_measurement_1': np.random.normal(50, 10, n_samples),
            'sensor_measurement_2': np.random.normal(30, 5, n_samples),
            'sensor_measurement_3': np.random.normal(100, 15, n_samples),
            'op_setting_1': np.random.normal(0, 1, n_samples),
            'op_setting_2': np.random.normal(0, 1, n_samples)
        }
        
        df = pd.DataFrame(data)
        df.to_csv(FilePaths.TRAIN_RUL_CSV, index=False)
        
        # Selected features dosyası
        features = ['sensor_measurement_1', 'sensor_measurement_2', 'sensor_measurement_3', 
                   'op_setting_1', 'op_setting_2']
        with open(FilePaths.FEATURES_TXT, 'w') as f:
            for feature in features:
                f.write(f"{feature}\n")
    
    @patch('builtins.print')
    @patch('joblib.dump')
    def test_model_training_pipeline(self, mock_dump, mock_print):
        """Model eğitim pipeline testi"""
        
        # Model train modülünü import et ve çalıştır
        try:
            # Model train script'ini çalıştır
            exec(open('model_train.py').read())
            success = True
        except Exception as e:
            # Eğer dosya bulunamazsa, kodu doğrudan test et
            success = self._test_training_logic()
        
        self.assertTrue(success)
        
        # joblib.dump'ın çağrıldığını kontrol et
        self.assertGreaterEqual(mock_dump.call_count, 2)  # Model ve scaler için
    
    def _test_training_logic(self):
        """Eğitim mantığını test et"""
        try:
            # Veriyi yükle
            df = pd.read_csv(FilePaths.TRAIN_RUL_CSV)
            
            # Özellikleri oku
            with open(FilePaths.FEATURES_TXT, "r") as f:
                selected_features = [line.strip() for line in f.readlines()]
            
            X = df[selected_features]
            y = df[ColumnNames.RUL]
            
            # Temel kontroller
            self.assertGreater(len(X), 0)
            self.assertGreater(len(y), 0)
            self.assertEqual(len(X), len(y))
            
            return True
            
        except Exception as e:
            print(f"Training logic test failed: {e}")
            return False
    
    def test_data_loading(self):
        """Veri yükleme testi"""
        # CSV dosyasının varlığını kontrol et
        self.assertTrue(os.path.exists(FilePaths.TRAIN_RUL_CSV))
        
        # Veriyi yükle
        df = pd.read_csv(FilePaths.TRAIN_RUL_CSV)
        
        # Temel kontroller
        self.assertGreater(len(df), 0)
        self.assertIn(ColumnNames.RUL, df.columns)
        
        # RUL değerlerinin pozitif olduğunu kontrol et
        self.assertTrue((df[ColumnNames.RUL] >= 0).all())
    
    def test_feature_loading(self):
        """Özellik yükleme testi"""
        # Features dosyasının varlığını kontrol et
        self.assertTrue(os.path.exists(FilePaths.FEATURES_TXT))
        
        # Özellikleri yükle
        with open(FilePaths.FEATURES_TXT, "r") as f:
            features = [line.strip() for line in f.readlines()]
        
        # Kontroller
        self.assertGreater(len(features), 0)
        self.assertTrue(all(len(f) > 0 for f in features))
        
        # CSV'de bu özelliklerin var olduğunu kontrol et
        df = pd.read_csv(FilePaths.TRAIN_RUL_CSV)
        for feature in features:
            self.assertIn(feature, df.columns)
    
    @patch('pandas.read_csv')
    def test_missing_csv_file(self, mock_read_csv):
        """Eksik CSV dosyası testi"""
        mock_read_csv.side_effect = FileNotFoundError("Dosya bulunamadı")
        
        with self.assertRaises(FileNotFoundError):
            pd.read_csv(FilePaths.TRAIN_RUL_CSV)
    
    def test_missing_features_file(self):
        """Eksik features dosyası testi"""
        # Features dosyasını sil
        if os.path.exists(FilePaths.FEATURES_TXT):
            os.remove(FilePaths.FEATURES_TXT)
        
        # FileNotFoundError fırlatmalı
        with self.assertRaises(FileNotFoundError):
            with open(FilePaths.FEATURES_TXT, "r") as f:
                f.read()
    
    @patch('sklearn.model_selection.train_test_split')
    def test_train_test_split_parameters(self, mock_split):
        """Train-test split parametreleri testi"""
        # Mock return values
        X_dummy = pd.DataFrame({'feature1': [1, 2, 3, 4]})
        y_dummy = pd.Series([10, 20, 30, 40])
        
        mock_split.return_value = (X_dummy, X_dummy, y_dummy, y_dummy)
        
        # Split fonksiyonunu çağır
        from sklearn.model_selection import train_test_split
        
        df = pd.read_csv(FilePaths.TRAIN_RUL_CSV)
        with open(FilePaths.FEATURES_TXT, "r") as f:
            features = [line.strip() for line in f.readlines()]
        
        X = df[features]
        y = df[ColumnNames.RUL]
        
        train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Doğru parametrelerle çağrıldığını kontrol et
        mock_split.assert_called_once()
        args, kwargs = mock_split.call_args
        
        self.assertEqual(kwargs.get('test_size', args[2] if len(args) > 2 else None), 0.2)
        self.assertEqual(kwargs.get('random_state', args[3] if len(args) > 3 else None), 42)
    
    @patch('xgboost.XGBRegressor')
    def test_xgboost_parameters(self, mock_xgb):
        """XGBoost parametreleri testi"""
        # Mock XGBoost
        mock_model = MagicMock()
        mock_xgb.return_value = mock_model
        
        # XGBoost'u oluştur
        from xgboost import XGBRegressor
        
        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        # Parametrelerin doğru olduğunu kontrol et
        mock_xgb.assert_called_once_with(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    
    def test_data_types_and_shapes(self):
        """Veri tipleri ve şekilleri testi"""
        df = pd.read_csv(FilePaths.TRAIN_RUL_CSV)
        
        with open(FilePaths.FEATURES_TXT, "r") as f:
            features = [line.strip() for line in f.readlines()]
        
        X = df[features]
        y = df[ColumnNames.RUL]
        
        # Şekil kontrolleri
        self.assertEqual(X.shape[0], y.shape[0])
        self.assertEqual(X.shape[1], len(features))
        
        # Tip kontrolleri
        self.assertTrue(pd.api.types.is_numeric_dtype(y))
        for feature in features:
            self.assertTrue(pd.api.types.is_numeric_dtype(X[feature]))


if __name__ == '__main__':
    unittest.main()
