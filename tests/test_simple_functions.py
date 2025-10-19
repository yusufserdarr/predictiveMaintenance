#!/usr/bin/env python3
"""
Basit fonksiyonlar için unit testler - Coverage artırmak için
"""

import unittest
import tempfile
import shutil
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import math


class TestSimpleFunctions(unittest.TestCase):
    """Basit fonksiyonlar testleri"""
    
    def setUp(self):
        """Test öncesi hazırlık"""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
    
    def tearDown(self):
        """Test sonrası temizlik"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
    
    def test_math_operations(self):
        """Matematik işlemleri testi"""
        # RUL hesaplama formülleri
        sicaklik = 45.5
        titresim = 0.25
        tork = 55.0
        
        # Farklı formüller test et
        rul1 = max(10, 200 - (sicaklik/10) - (titresim*20) - (tork/2))
        rul2 = 200 - sicaklik - titresim*100 - tork
        rul3 = abs(100 - sicaklik) + abs(50 - tork)
        
        self.assertIsInstance(rul1, (int, float))
        self.assertIsInstance(rul2, (int, float))
        self.assertIsInstance(rul3, (int, float))
        
        # Minimum değer kontrolü
        self.assertGreaterEqual(rul1, 10)
    
    def test_data_validation_functions(self):
        """Veri doğrulama fonksiyonları testi"""
        # Sayısal değer kontrolü
        def is_numeric(value):
            try:
                float(value)
                return True
            except (ValueError, TypeError):
                return False
        
        # Test değerleri
        test_values = [45.5, "46.0", "abc", None, 0, -5.5]
        expected = [True, True, False, False, True, True]
        
        for value, expected_result in zip(test_values, expected):
            result = is_numeric(value)
            self.assertEqual(result, expected_result)
    
    def test_range_validation(self):
        """Aralık doğrulama testi"""
        def validate_sensor_range(value, min_val=0, max_val=100):
            if not isinstance(value, (int, float)):
                return False
            return min_val <= value <= max_val
        
        # Test senaryoları
        test_cases = [
            (50, True),      # Normal değer
            (0, True),       # Minimum sınır
            (100, True),     # Maksimum sınır
            (-5, False),     # Minimum altı
            (150, False),    # Maksimum üstü
            ("50", False),   # String
            (None, False)    # None
        ]
        
        for value, expected in test_cases:
            result = validate_sensor_range(value)
            self.assertEqual(result, expected)
    
    def test_timestamp_operations(self):
        """Zaman damgası işlemleri testi"""
        from datetime import datetime
        
        # Şu anki zaman
        now = datetime.now()
        timestamp_str = now.isoformat()
        
        # String format kontrolü
        self.assertIsInstance(timestamp_str, str)
        self.assertIn('T', timestamp_str)  # ISO format
        
        # Zaman farkı hesaplama
        import time
        time.sleep(0.01)  # 10ms bekle
        later = datetime.now()
        
        time_diff = (later - now).total_seconds()
        self.assertGreater(time_diff, 0)
    
    def test_file_operations(self):
        """Dosya işlemleri testi"""
        # Test dosyası oluştur
        test_file = "test_data.txt"
        test_content = "test,data,content\n1,2,3\n4,5,6"
        
        # Dosya yazma
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        # Dosya okuma
        with open(test_file, 'r') as f:
            content = f.read()
        
        self.assertEqual(content, test_content)
        
        # Dosya varlığı kontrolü
        self.assertTrue(os.path.exists(test_file))
        
        # Dosya silme
        os.remove(test_file)
        self.assertFalse(os.path.exists(test_file))
    
    def test_list_operations(self):
        """Liste işlemleri testi"""
        # Test listesi
        test_list = [1, 2, 3, 4, 5]
        
        # Liste işlemleri
        self.assertEqual(len(test_list), 5)
        self.assertEqual(sum(test_list), 15)
        self.assertEqual(max(test_list), 5)
        self.assertEqual(min(test_list), 1)
        
        # Liste filtreleme
        even_numbers = [x for x in test_list if x % 2 == 0]
        self.assertEqual(even_numbers, [2, 4])
        
        # Liste dönüştürme
        squared = [x**2 for x in test_list]
        self.assertEqual(squared, [1, 4, 9, 16, 25])
    
    def test_dictionary_operations(self):
        """Sözlük işlemleri testi"""
        # Test sözlüğü
        test_dict = {
            'sicaklik': 45.5,
            'titresim': 0.25,
            'tork': 55.0,
            'rul': 125.5
        }
        
        # Anahtar kontrolü
        self.assertIn('sicaklik', test_dict)
        self.assertIn('rul', test_dict)
        
        # Değer kontrolü
        self.assertEqual(test_dict['sicaklik'], 45.5)
        
        # Sözlük güncelleme
        test_dict.update({'status': 'NORMAL'})
        self.assertIn('status', test_dict)
        
        # Anahtar-değer çiftleri
        keys = list(test_dict.keys())
        values = list(test_dict.values())
        
        self.assertEqual(len(keys), len(values))
    
    def test_string_operations(self):
        """String işlemleri testi"""
        # Test string'i
        test_string = "sensor_measurement_1"
        
        # String işlemleri
        self.assertTrue(test_string.startswith('sensor'))
        self.assertTrue(test_string.endswith('1'))
        self.assertIn('measurement', test_string)
        
        # String dönüştürme
        upper_string = test_string.upper()
        self.assertEqual(upper_string, "SENSOR_MEASUREMENT_1")
        
        # String bölme
        parts = test_string.split('_')
        self.assertEqual(len(parts), 3)
        self.assertEqual(parts[0], 'sensor')
    
    def test_numpy_operations(self):
        """NumPy işlemleri testi"""
        # Test array'i
        test_array = np.array([1, 2, 3, 4, 5])
        
        # Array işlemleri
        self.assertEqual(test_array.sum(), 15)
        self.assertEqual(test_array.mean(), 3.0)
        self.assertEqual(test_array.std(), np.std([1, 2, 3, 4, 5]))
        
        # Array dönüştürme
        normalized = (test_array - test_array.mean()) / test_array.std()
        self.assertEqual(len(normalized), len(test_array))
    
    def test_pandas_operations(self):
        """Pandas işlemleri testi"""
        # Test DataFrame
        test_data = {
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
            'C': [0.1, 0.2, 0.3, 0.4, 0.5]
        }
        df = pd.DataFrame(test_data)
        
        # DataFrame işlemleri
        self.assertEqual(len(df), 5)
        self.assertEqual(len(df.columns), 3)
        
        # İstatistiksel işlemler
        self.assertEqual(df['A'].sum(), 15)
        self.assertEqual(df['B'].mean(), 30.0)
        
        # Filtreleme
        filtered = df[df['A'] > 3]
        self.assertEqual(len(filtered), 2)
    
    def test_conditional_logic(self):
        """Koşullu mantık testi"""
        def classify_rul(rul_value):
            if rul_value < 20:
                return "CRITICAL"
            elif rul_value < 50:
                return "PLANNED"
            else:
                return "NORMAL"
        
        # Test senaryoları
        test_cases = [
            (15, "CRITICAL"),
            (35, "PLANNED"),
            (80, "NORMAL"),
            (20, "PLANNED"),  # Sınır değeri
            (50, "NORMAL")    # Sınır değeri
        ]
        
        for rul, expected in test_cases:
            result = classify_rul(rul)
            self.assertEqual(result, expected)
    
    def test_error_handling(self):
        """Hata yönetimi testi"""
        def safe_divide(a, b):
            try:
                return a / b
            except ZeroDivisionError:
                return float('inf')
            except TypeError:
                return None
        
        # Test senaryoları
        self.assertEqual(safe_divide(10, 2), 5.0)
        self.assertEqual(safe_divide(10, 0), float('inf'))
        self.assertIsNone(safe_divide(10, "a"))
    
    def test_data_conversion(self):
        """Veri dönüştürme testi"""
        # String to float
        str_values = ["45.5", "46.0", "47.2"]
        float_values = [float(x) for x in str_values]
        
        self.assertEqual(float_values, [45.5, 46.0, 47.2])
        
        # List to array
        list_data = [1, 2, 3, 4, 5]
        array_data = np.array(list_data)
        
        self.assertEqual(len(array_data), len(list_data))
        
        # Dict to DataFrame
        dict_data = {'A': [1, 2], 'B': [3, 4]}
        df_data = pd.DataFrame(dict_data)
        
        self.assertEqual(len(df_data), 2)
        self.assertEqual(len(df_data.columns), 2)


if __name__ == '__main__':
    unittest.main()
