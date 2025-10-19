#!/usr/bin/env python3
"""
RUL Generator modülü için unit testler
"""

import unittest
import tempfile
import shutil
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from constants import ColumnNames, FilePaths


class TestRulGenerator(unittest.TestCase):
    """RUL Generator testleri"""
    
    def setUp(self):
        """Test öncesi hazırlık"""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        # Test verisi oluştur
        self.create_test_input_data()
    
    def tearDown(self):
        """Test sonrası temizlik"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
    
    def create_test_input_data(self):
        """Test için input verisi oluştur"""
        # train.txt formatında veri oluştur
        np.random.seed(42)
        
        data = []
        for unit in range(1, 4):  # 3 ünite
            for cycle in range(1, 21):  # Her ünite 20 cycle
                row = [unit, cycle]  # unit_number, time_in_cycles
                row.extend([0.1, 0.2, 0.3])  # op_settings
                row.extend(np.random.normal(50, 10, 21))  # 21 sensör
                data.append(row)
        
        # Space-separated format'ta kaydet
        with open("train.txt", "w") as f:
            for row in data:
                f.write(" ".join(map(str, row)) + "\n")
    
    @patch('builtins.print')
    def test_rul_generation_pipeline(self, mock_print):
        """RUL oluşturma pipeline testi"""
        
        try:
            # RUL generator script'ini çalıştır
            exec(open('rul_generator.py').read())
            success = True
        except Exception as e:
            print(f"RUL generation failed: {e}")
            success = False
        
        self.assertTrue(success)
        
        # Çıktı dosyasının oluştuğunu kontrol et
        self.assertTrue(os.path.exists(FilePaths.TRAIN_RUL_CSV))
        
        # Print çağrısını kontrol et
        mock_print.assert_called()
    
    def test_output_file_structure(self):
        """Çıktı dosyası yapısı testi"""
        # RUL generator'ı çalıştır
        exec(open('../rul_generator.py').read())
        
        # Çıktı dosyasını oku
        df = pd.read_csv(FilePaths.TRAIN_RUL_CSV)
        
        # Temel kontroller
        self.assertGreater(len(df), 0)
        self.assertIn(ColumnNames.UNIT_NUMBER, df.columns)
        self.assertIn(ColumnNames.TIME_IN_CYCLES, df.columns)
        self.assertIn(ColumnNames.RUL, df.columns)
        
        # RUL hesaplamasını kontrol et
        for unit in df[ColumnNames.UNIT_NUMBER].unique():
            unit_data = df[df[ColumnNames.UNIT_NUMBER] == unit]
            max_cycle = unit_data[ColumnNames.TIME_IN_CYCLES].max()
            
            # Her cycle için RUL = max_cycle - current_cycle olmalı
            for _, row in unit_data.iterrows():
                expected_rul = max_cycle - row[ColumnNames.TIME_IN_CYCLES]
                self.assertEqual(row[ColumnNames.RUL], expected_rul)
    
    def test_column_names_generation(self):
        """Kolon isimleri oluşturma testi"""
        # RUL generator'ı çalıştır
        exec(open('../rul_generator.py').read())
        
        df = pd.read_csv(FilePaths.TRAIN_RUL_CSV)
        
        # Beklenen kolonlar
        expected_cols = [
            ColumnNames.UNIT_NUMBER,
            ColumnNames.TIME_IN_CYCLES,
            "op_setting_1", "op_setting_2", "op_setting_3"
        ]
        expected_cols.extend([f"sensor_measurement_{i}" for i in range(1, 22)])
        expected_cols.append(ColumnNames.RUL)
        
        # Tüm beklenen kolonların var olduğunu kontrol et
        for col in expected_cols:
            self.assertIn(col, df.columns)
    
    def test_rul_calculation_logic(self):
        """RUL hesaplama mantığı testi"""
        # RUL generator'ı çalıştır
        exec(open('../rul_generator.py').read())
        
        df = pd.read_csv(FilePaths.TRAIN_RUL_CSV)
        
        # Her ünite için kontrol
        for unit in df[ColumnNames.UNIT_NUMBER].unique():
            unit_data = df[df[ColumnNames.UNIT_NUMBER] == unit].sort_values(ColumnNames.TIME_IN_CYCLES)
            
            # RUL değerleri azalan sırada olmalı
            rul_values = unit_data[ColumnNames.RUL].values
            self.assertTrue(all(rul_values[i] >= rul_values[i+1] for i in range(len(rul_values)-1)))
            
            # Son cycle'da RUL = 0 olmalı
            self.assertEqual(rul_values[-1], 0)
            
            # İlk cycle'da RUL = max_cycle - 1 olmalı
            max_cycle = unit_data[ColumnNames.TIME_IN_CYCLES].max()
            first_rul = unit_data[ColumnNames.RUL].iloc[0]
            self.assertEqual(first_rul, max_cycle - 1)
    
    def test_data_types(self):
        """Veri tipleri testi"""
        # RUL generator'ı çalıştır
        exec(open('../rul_generator.py').read())
        
        df = pd.read_csv(FilePaths.TRAIN_RUL_CSV)
        
        # Sayısal kolonların tiplerini kontrol et
        numeric_cols = [ColumnNames.UNIT_NUMBER, ColumnNames.TIME_IN_CYCLES, ColumnNames.RUL]
        numeric_cols.extend([f"sensor_measurement_{i}" for i in range(1, 22)])
        numeric_cols.extend([f"op_setting_{i}" for i in range(1, 4)])
        
        for col in numeric_cols:
            self.assertTrue(pd.api.types.is_numeric_dtype(df[col]), f"{col} should be numeric")
    
    def test_missing_input_file(self):
        """Eksik input dosyası testi"""
        # train.txt dosyasını sil
        if os.path.exists("train.txt"):
            os.remove("train.txt")
        
        # FileNotFoundError fırlatmalı
        with self.assertRaises(FileNotFoundError):
            exec(open('rul_generator.py').read())
    
    @patch('pandas.read_csv')
    def test_empty_input_file(self, mock_read_csv):
        """Boş input dosyası testi"""
        # Boş DataFrame döndür
        mock_read_csv.return_value = pd.DataFrame()
        
        # Hata fırlatmalı
        with self.assertRaises((ValueError, KeyError)):
            exec(open('rul_generator.py').read())
    
    def test_merge_operation(self):
        """Merge işlemi testi"""
        # RUL generator'ı çalıştır
        exec(open('../rul_generator.py').read())
        
        df = pd.read_csv(FilePaths.TRAIN_RUL_CSV)
        
        # Her satırda unit_number ve RUL olmalı
        self.assertFalse(df[ColumnNames.UNIT_NUMBER].isna().any())
        self.assertFalse(df[ColumnNames.RUL].isna().any())
        
        # Merge sonrası satır sayısı korunmalı
        original_units = df[ColumnNames.UNIT_NUMBER].nunique()
        self.assertGreaterEqual(original_units, 1)
    
    def test_max_cycle_column_removal(self):
        """Max cycle kolonunun kaldırılması testi"""
        # RUL generator'ı çalıştır
        exec(open('../rul_generator.py').read())
        
        df = pd.read_csv(FilePaths.TRAIN_RUL_CSV)
        
        # max_cycle kolonu olmamalı (kaldırılmış olmalı)
        self.assertNotIn(ColumnNames.MAX_CYCLE, df.columns)
    
    def test_output_file_creation_message(self):
        """Çıktı dosyası oluşturma mesajı testi"""
        with patch('builtins.print') as mock_print:
            # RUL generator'ı çalıştır
            exec(open('rul_generator.py').read())
            
            # Print çağrılarını kontrol et
            print_calls = [str(call) for call in mock_print.call_args_list]
            
            # Dosya oluşturma mesajını ara
            found_message = any(FilePaths.TRAIN_RUL_CSV in call for call in print_calls)
            self.assertTrue(found_message)


if __name__ == '__main__':
    unittest.main()
