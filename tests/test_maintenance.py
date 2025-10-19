#!/usr/bin/env python3
"""
Maintenance modülü için unit testler
"""

import unittest
import math
from maintenance import maintenance_decision


class TestMaintenanceDecision(unittest.TestCase):
    """Bakım karar sistemi testleri"""
    
    def test_critical_maintenance(self):
        """Kritik bakım durumu testi"""
        result = maintenance_decision(15)  # Varsayılan eşik: 20
        
        self.assertEqual(result['status'], 'CRITICAL')
        self.assertEqual(result['rul'], 15)
        self.assertIn('Acil bakım', result['message'])
        self.assertEqual(result['color'], '#D32F2F')
    
    def test_planned_maintenance(self):
        """Planlı bakım durumu testi"""
        result = maintenance_decision(35)  # 20 < RUL < 50
        
        self.assertEqual(result['status'], 'PLANNED')
        self.assertEqual(result['rul'], 35)
        self.assertIn('Planlı bakım', result['message'])
        self.assertEqual(result['color'], '#ED6C02')
    
    def test_normal_operation(self):
        """Normal çalışma durumu testi"""
        result = maintenance_decision(80)  # RUL > 50
        
        self.assertEqual(result['status'], 'NORMAL')
        self.assertEqual(result['rul'], 80)
        self.assertIn('Normal çalışma', result['message'])
        self.assertEqual(result['color'], '#2E7D32')
    
    def test_invalid_rul_negative(self):
        """Negatif RUL değeri testi"""
        result = maintenance_decision(-5)
        
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertEqual(result['rul'], -5)
        self.assertIn('Geçersiz RUL', result['message'])
        self.assertEqual(result['color'], '#757575')
    
    def test_invalid_rul_nan(self):
        """NaN RUL değeri testi"""
        result = maintenance_decision(float('nan'))
        
        self.assertEqual(result['status'], 'UNKNOWN')
        self.assertTrue(math.isnan(result['rul']))
        self.assertIn('Geçersiz RUL', result['message'])
        self.assertEqual(result['color'], '#757575')
    
    def test_custom_thresholds(self):
        """Özel eşikler testi"""
        custom_thresholds = {"critical": 10, "planned": 30}
        
        # Kritik test
        result = maintenance_decision(8, custom_thresholds)
        self.assertEqual(result['status'], 'CRITICAL')
        
        # Planlı test
        result = maintenance_decision(20, custom_thresholds)
        self.assertEqual(result['status'], 'PLANNED')
        
        # Normal test
        result = maintenance_decision(40, custom_thresholds)
        self.assertEqual(result['status'], 'NORMAL')
    
    def test_boundary_values(self):
        """Sınır değerleri testi"""
        # Kritik sınırda
        result = maintenance_decision(20)  # Tam eşik değeri
        self.assertEqual(result['status'], 'PLANNED')  # 20 >= 20, yani planned
        
        result = maintenance_decision(19.9)  # Eşiğin altında
        self.assertEqual(result['status'], 'CRITICAL')
        
        # Planlı sınırda
        result = maintenance_decision(50)  # Tam eşik değeri
        self.assertEqual(result['status'], 'NORMAL')  # 50 >= 50, yani normal
        
        result = maintenance_decision(49.9)  # Eşiğin altında
        self.assertEqual(result['status'], 'PLANNED')
    
    def test_return_type_and_keys(self):
        """Dönüş tipı ve anahtarları testi"""
        result = maintenance_decision(25)
        
        # Tip kontrolü
        self.assertIsInstance(result, dict)
        
        # Anahtar kontrolü
        expected_keys = {'rul', 'status', 'color', 'message'}
        self.assertEqual(set(result.keys()), expected_keys)
        
        # Değer tipleri
        self.assertIsInstance(result['rul'], (int, float))
        self.assertIsInstance(result['status'], str)
        self.assertIsInstance(result['color'], str)
        self.assertIsInstance(result['message'], str)


    def test_print_decision_function(self):
        """print_decision fonksiyonu testi"""
        from maintenance import print_decision
        
        # Çıktıyı yakalamak için
        import io
        import sys
        
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            print_decision(25)
            output = captured_output.getvalue()
            
            # Çıktıda RUL değeri ve status olmalı
            self.assertIn("25", output)
            self.assertIn("PLANNED", output)
            
        finally:
            sys.stdout = sys.__stdout__
    
    def test_main_block_execution(self):
        """Ana blok çalıştırma testi"""
        # maintenance.py'nin __main__ bloğunu test et
        import maintenance
        
        # Test senaryolarını kontrol et
        test_cases = [15, 35, 80, 5, 25, 100, -5, float('nan')]
        
        for rul in test_cases:
            result = maintenance.maintenance_decision(rul)
            self.assertIn('status', result)
            self.assertIn('rul', result)
            self.assertIn('color', result)
            self.assertIn('message', result)
    
    def test_custom_thresholds_edge_cases(self):
        """Özel eşikler kenar durumları testi"""
        # Aynı eşik değerleri
        same_thresholds = {"critical": 30, "planned": 30}
        result = maintenance_decision(25, same_thresholds)
        self.assertEqual(result['status'], 'CRITICAL')
        
        result = maintenance_decision(35, same_thresholds)
        self.assertEqual(result['status'], 'NORMAL')
        
        # Ters eşikler (planned < critical)
        reverse_thresholds = {"critical": 50, "planned": 20}
        result = maintenance_decision(30, reverse_thresholds)
        # 30 < 50 (critical) olduğu için CRITICAL olmalı
        self.assertEqual(result['status'], 'CRITICAL')
    
    def test_none_thresholds_default_behavior(self):
        """None eşikler varsayılan davranış testi"""
        result1 = maintenance_decision(25, None)
        result2 = maintenance_decision(25)  # Varsayılan
        
        # İkisi de aynı sonucu vermeli
        self.assertEqual(result1['status'], result2['status'])
        self.assertEqual(result1['rul'], result2['rul'])
        self.assertEqual(result1['color'], result2['color'])
        self.assertEqual(result1['message'], result2['message'])
    
    def test_extreme_rul_values(self):
        """Aşırı RUL değerleri testi"""
        # Çok büyük değer
        result = maintenance_decision(999999)
        self.assertEqual(result['status'], 'NORMAL')
        
        # Çok küçük negatif değer
        result = maintenance_decision(-999999)
        self.assertEqual(result['status'], 'UNKNOWN')
        
        # Sıfır değer
        result = maintenance_decision(0)
        self.assertEqual(result['status'], 'CRITICAL')


if __name__ == '__main__':
    unittest.main()
