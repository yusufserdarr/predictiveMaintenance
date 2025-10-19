#!/usr/bin/env python3
"""
Reporting modülü için unit testler
"""

import unittest
import os
import tempfile
import shutil
import csv
from datetime import datetime
from unittest.mock import patch, mock_open
from reporting import ensure_dirs, append_prediction_log


class TestReporting(unittest.TestCase):
    """Reporting modülü testleri"""
    
    def setUp(self):
        """Test öncesi hazırlık"""
        self.test_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.test_dir)
    
    def tearDown(self):
        """Test sonrası temizlik"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.test_dir)
    
    def test_ensure_dirs_creates_directories(self):
        """ensure_dirs klasör oluşturma testi"""
        # Klasörler yokken
        self.assertFalse(os.path.exists('logs'))
        self.assertFalse(os.path.exists('reports'))
        
        # Fonksiyonu çalıştır
        ensure_dirs()
        
        # Klasörlerin oluştuğunu kontrol et
        self.assertTrue(os.path.exists('logs'))
        self.assertTrue(os.path.exists('reports'))
        self.assertTrue(os.path.isdir('logs'))
        self.assertTrue(os.path.isdir('reports'))
    
    def test_ensure_dirs_existing_directories(self):
        """ensure_dirs mevcut klasörler testi"""
        # Önce klasörleri oluştur
        os.makedirs('logs')
        os.makedirs('reports')
        
        # Fonksiyonu çalıştır (hata vermemeli)
        try:
            ensure_dirs()
            success = True
        except Exception:
            success = False
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists('logs'))
        self.assertTrue(os.path.exists('reports'))
    
    def test_append_prediction_log_new_file(self):
        """append_prediction_log yeni dosya testi"""
        ensure_dirs()
        
        log_data = {
            'timestamp': '2024-01-01T12:00:00',
            'sicaklik': 45.5,
            'titresim': 0.25,
            'tork': 55.0,
            'rul': 125.5,
            'status': 'NORMAL'
        }
        
        # Log dosyası yokken
        log_path = 'logs/predictions.csv'
        self.assertFalse(os.path.exists(log_path))
        
        # Log yaz
        append_prediction_log(log_data)
        
        # Dosyanın oluştuğunu kontrol et
        self.assertTrue(os.path.exists(log_path))
        
        # İçeriği kontrol et
        with open(log_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['timestamp'], '2024-01-01T12:00:00')
        self.assertEqual(float(rows[0]['sicaklik']), 45.5)
        self.assertEqual(rows[0]['status'], 'NORMAL')
    
    def test_append_prediction_log_existing_file(self):
        """append_prediction_log mevcut dosya testi"""
        ensure_dirs()
        
        # İlk log
        log_data1 = {
            'timestamp': '2024-01-01T12:00:00',
            'sicaklik': 45.5,
            'titresim': 0.25,
            'tork': 55.0,
            'rul': 125.5,
            'status': 'NORMAL'
        }
        
        # İkinci log
        log_data2 = {
            'timestamp': '2024-01-01T13:00:00',
            'sicaklik': 48.0,
            'titresim': 0.30,
            'tork': 58.0,
            'rul': 115.0,
            'status': 'PLANNED'
        }
        
        # İki log yaz
        append_prediction_log(log_data1)
        append_prediction_log(log_data2)
        
        # İçeriği kontrol et
        log_path = 'logs/predictions.csv'
        with open(log_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['status'], 'NORMAL')
        self.assertEqual(rows[1]['status'], 'PLANNED')
    
    def test_append_prediction_log_data_types(self):
        """append_prediction_log veri tipleri testi"""
        ensure_dirs()
        
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'sicaklik': 45,  # int
            'titresim': 0.25,  # float
            'tork': '55',  # string
            'rul': 125.5,
            'status': 'NORMAL'
        }
        
        # Hata vermeden yazabilmeli
        try:
            append_prediction_log(log_data)
            success = True
        except Exception as e:
            success = False
            print(f"Hata: {e}")
        
        self.assertTrue(success)
    
    @patch('builtins.open', side_effect=OSError("Dosya yazma hatası"))
    def test_append_prediction_log_error_handling(self, mock_open):
        """append_prediction_log hata yönetimi testi"""
        log_data = {
            'timestamp': '2024-01-01T12:00:00',
            'sicaklik': 45.5,
            'titresim': 0.25,
            'tork': 55.0,
            'rul': 125.5,
            'status': 'NORMAL'
        }
        
        # OSError fırlatmalı
        with self.assertRaises(OSError):
            append_prediction_log(log_data)
    
    def test_log_data_validation(self):
        """Log verisi doğrulama testi"""
        ensure_dirs()
        
        # Eksik anahtar
        incomplete_data = {
            'timestamp': '2024-01-01T12:00:00',
            'sicaklik': 45.5
            # Diğer alanlar eksik
        }
        
        # ValueError fırlatmalı (eksik alanlar için)
        with self.assertRaises(ValueError):
            append_prediction_log(incomplete_data)


if __name__ == '__main__':
    unittest.main()
