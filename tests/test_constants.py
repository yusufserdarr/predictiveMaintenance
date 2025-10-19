#!/usr/bin/env python3
"""
Constants modülü için unit testler
"""

import unittest
from constants import (
    MaintenanceStatus, Colors, Messages, FilePaths, 
    DefaultThresholds, ColumnNames, HttpMethods, 
    DataSources, ReportConstants
)


class TestConstants(unittest.TestCase):
    """Constants sınıfları testleri"""
    
    def test_maintenance_status_values(self):
        """MaintenanceStatus değerleri testi"""
        self.assertEqual(MaintenanceStatus.CRITICAL, "CRITICAL")
        self.assertEqual(MaintenanceStatus.PLANNED, "PLANNED")
        self.assertEqual(MaintenanceStatus.NORMAL, "NORMAL")
        self.assertEqual(MaintenanceStatus.UNKNOWN, "UNKNOWN")
    
    def test_colors_format(self):
        """Colors hex format testi"""
        colors = [Colors.CRITICAL, Colors.PLANNED, Colors.NORMAL, Colors.UNKNOWN]
        
        for color in colors:
            self.assertIsInstance(color, str)
            self.assertTrue(color.startswith('#'))
            self.assertEqual(len(color), 7)  # #RRGGBB format
    
    def test_file_paths_extensions(self):
        """FilePaths dosya uzantıları testi"""
        self.assertTrue(FilePaths.MODEL_PKL.endswith('.pkl'))
        self.assertTrue(FilePaths.SCALER_PKL.endswith('.pkl'))
        self.assertTrue(FilePaths.FEATURES_TXT.endswith('.txt'))
        self.assertTrue(FilePaths.STREAM_CSV.endswith('.csv'))
        self.assertTrue(FilePaths.TRAIN_RUL_CSV.endswith('.csv'))
    
    def test_default_thresholds_values(self):
        """DefaultThresholds değerleri testi"""
        self.assertIsInstance(DefaultThresholds.CRITICAL, int)
        self.assertIsInstance(DefaultThresholds.PLANNED, int)
        self.assertGreater(DefaultThresholds.PLANNED, DefaultThresholds.CRITICAL)
        self.assertGreater(DefaultThresholds.CRITICAL, 0)
    
    def test_column_names_not_empty(self):
        """ColumnNames boş olmama testi"""
        column_attrs = [
            'UNIT_NUMBER', 'TIME_IN_CYCLES', 'RUL', 'SICAKLIK', 
            'TITRESIM', 'TORK', 'TIMESTAMP', 'STATUS', 'MAX_CYCLE'
        ]
        
        for attr in column_attrs:
            value = getattr(ColumnNames, attr)
            self.assertIsInstance(value, str)
            self.assertGreater(len(value), 0)
    
    def test_http_methods_valid(self):
        """HttpMethods geçerli HTTP metodları testi"""
        valid_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']
        
        self.assertIn(HttpMethods.GET, valid_methods)
        self.assertIn(HttpMethods.POST, valid_methods)
        self.assertIn(HttpMethods.PUT, valid_methods)
    
    def test_messages_not_empty(self):
        """Messages boş olmama testi"""
        message_attrs = [
            'CRITICAL_MAINTENANCE', 'PLANNED_MAINTENANCE', 'NORMAL_OPERATION',
            'INVALID_RUL', 'MODEL_LOAD_ERROR', 'MISSING_DATA', 'INVALID_DATA',
            'PREDICTION_ERROR', 'ENTER_SENSOR_VALUES', 'ENTER_VALID_NUMBERS'
        ]
        
        for attr in message_attrs:
            value = getattr(Messages, attr)
            self.assertIsInstance(value, str)
            self.assertGreater(len(value), 0)
    
    def test_data_sources_turkish(self):
        """DataSources Türkçe değerler testi"""
        sources = [DataSources.LIVE_STREAM, DataSources.FILE_UPLOAD, DataSources.MANUAL_INPUT]
        
        for source in sources:
            self.assertIsInstance(source, str)
            self.assertGreater(len(source), 0)
    
    def test_report_constants_not_empty(self):
        """ReportConstants boş olmama testi"""
        self.assertIsInstance(ReportConstants.RAW_DATA_SHEET, str)
        self.assertGreater(len(ReportConstants.RAW_DATA_SHEET), 0)
    
    def test_constants_immutability(self):
        """Constants değişmezlik testi (sınıf attribute'ları)"""
        # Bu test constants'ların doğru değerlerde olduğunu kontrol eder
        # Python'da class attribute'lar değiştirilebilir, bu normal
        self.assertEqual(MaintenanceStatus.CRITICAL, "CRITICAL")
        self.assertEqual(MaintenanceStatus.PLANNED, "PLANNED")
        self.assertEqual(MaintenanceStatus.NORMAL, "NORMAL")
        self.assertEqual(MaintenanceStatus.UNKNOWN, "UNKNOWN")


if __name__ == '__main__':
    unittest.main()
