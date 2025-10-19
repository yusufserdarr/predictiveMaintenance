# main_gui.py
import sys
import pandas as pd
import joblib
import os
import datetime
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtChart import QChart, QChartView, QLineSeries, QValueAxis, QDateTimeAxis
from maintenance import maintenance_decision
from reporting import append_prediction_log, daily_report_to_excel, ensure_dirs
from lime_explain import explain_instance, open_html_in_browser
from shap_analysis import shap_summary_png, shap_local_png, load_sample_data
from constants import FilePaths, Messages, ColumnNames
import subprocess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ======= AYARLAR =======
# Constants dosyasından alınıyor
STREAM_CSV = "logs/stream.csv"
# =======================

class PredictionApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 Predictive Maintenance Dashboard - Professional Edition")
        self.setGeometry(100, 50, 1400, 900)
        
        # Modern stil uygula
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #2c3e50;
            }
            QPushButton {
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
                min-height: 25px;
            }
            QLineEdit, QDoubleSpinBox {
                border: 2px solid #bdc3c7;
                border-radius: 4px;
                padding: 5px;
                background-color: white;
            }
            QLineEdit:focus, QDoubleSpinBox:focus {
                border-color: #3498db;
            }
            QLabel {
                color: #2c3e50;
            }
        """)

        # Raporlama klasörlerini hazırla
        ensure_dirs()

        # Model ve scaler yükle
        try:
            self.model = joblib.load(FilePaths.MODEL_PKL)
            self.scaler = joblib.load(FilePaths.SCALER_PKL)
            
            # Özellikleri oku
            with open(FilePaths.FEATURES_TXT, "r") as f:
                self.features = [line.strip() for line in f.readlines()]
                
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, Messages.MODEL_LOAD_ERROR, f"Model dosyaları yüklenemedi:\n{str(e)}")
            sys.exit(1)

        # Canlı akış için timer
        self.stream_timer = QtCore.QTimer()
        self.stream_timer.timeout.connect(self.update_from_stream)
        self.is_streaming = False
        self.last_row_count = 0
        
        # Tahmin geçmişi
        self.prediction_history = []
        
        # Sensör input'ları için dictionary
        self.sensor_inputs = {}

        # Arayüzü oluştur
        self.init_ui()

    def init_ui(self):
        # Ana widget ve layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # Ana horizontal layout (sol panel + sağ panel)
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        
        # Sol panel (kontroller)
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Sağ panel (grafikler ve sonuçlar)
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 2)
        
        # Status bar
        self.statusBar().showMessage("Hazır - Predictive Maintenance Dashboard")
        
        # Menu bar
        self.create_menu_bar()

    def create_menu_bar(self):
        """Modern menu bar oluştur"""
        menubar = self.menuBar()
        
        # Dosya menüsü
        file_menu = menubar.addMenu('📁 Dosya')
        
        export_action = QtWidgets.QAction('📊 Rapor Dışa Aktar', self)
        export_action.triggered.connect(self.download_daily_report)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QtWidgets.QAction('❌ Çıkış', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Analiz menüsü
        analysis_menu = menubar.addMenu('🔬 Analiz')
        
        lime_action = QtWidgets.QAction('🔍 LIME Açıklaması', self)
        lime_action.triggered.connect(self.show_lime_explanation)
        analysis_menu.addAction(lime_action)
        
        shap_summary_action = QtWidgets.QAction('📊 SHAP Özet', self)
        shap_summary_action.triggered.connect(self.show_shap_summary)
        analysis_menu.addAction(shap_summary_action)
        
        shap_local_action = QtWidgets.QAction('🎯 SHAP Lokal', self)
        shap_local_action.triggered.connect(self.show_shap_local)
        analysis_menu.addAction(shap_local_action)

    def create_left_panel(self):
        """Sol kontrol panelini oluştur"""
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        
        # Başlık
        title = QtWidgets.QLabel("🧠 Predictive Maintenance")
        title.setFont(QtGui.QFont("Arial", 18, QtGui.QFont.Bold))
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("color: #2c3e50; margin: 10px; padding: 10px;")
        left_layout.addWidget(title)
        
        # Canlı akış kontrolleri
        stream_group = self.create_stream_controls()
        left_layout.addWidget(stream_group)
        
        # Sensör girişleri
        sensor_group = self.create_sensor_inputs()
        left_layout.addWidget(sensor_group)
        
        # Ana işlem butonları
        action_group = self.create_action_buttons()
        left_layout.addWidget(action_group)
        
        # Tahmin sonuçları
        result_group = self.create_result_display()
        left_layout.addWidget(result_group)
        
        left_layout.addStretch()
        return left_widget

    def create_stream_controls(self):
        """Canlı akış kontrol grubu"""
        group = QtWidgets.QGroupBox("🔴 Canlı Veri Akışı")
        layout = QtWidgets.QVBoxLayout()
        
        # Durum göstergesi
        self.stream_status = QtWidgets.QLabel("⚪ Akış Durumu: Durduruldu")
        self.stream_status.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(self.stream_status)
        
        # Butonlar
        button_layout = QtWidgets.QHBoxLayout()
        
        self.start_stream_btn = QtWidgets.QPushButton("▶️ Başlat")
        self.start_stream_btn.setStyleSheet("background-color: #27ae60; color: white;")
        self.start_stream_btn.clicked.connect(self.start_stream)
        
        self.stop_stream_btn = QtWidgets.QPushButton("⏹️ Durdur")
        self.stop_stream_btn.setStyleSheet("background-color: #e74c3c; color: white;")
        self.stop_stream_btn.clicked.connect(self.stop_stream)
        self.stop_stream_btn.setEnabled(False)
        
        button_layout.addWidget(self.start_stream_btn)
        button_layout.addWidget(self.stop_stream_btn)
        layout.addLayout(button_layout)
        
        group.setLayout(layout)
        return group

    def create_sensor_inputs(self):
        """Tüm sensör girişlerini oluştur"""
        group = QtWidgets.QGroupBox("🌡️ Sensör Değerleri (10 Sensör)")
        layout = QtWidgets.QVBoxLayout()
        
        # Ana sensörler
        main_group = QtWidgets.QGroupBox("🔧 Ana Sensörler")
        main_layout = QtWidgets.QFormLayout()
        
        # Sıcaklık
        self.sicaklik_input = QtWidgets.QDoubleSpinBox()
        self.sicaklik_input.setRange(200.0, 700.0)
        self.sicaklik_input.setValue(450.0)
        self.sicaklik_input.setSuffix(" °C")
        main_layout.addRow("🌡️ Sıcaklık:", self.sicaklik_input)
        
        # Titreşim
        self.titresim_input = QtWidgets.QDoubleSpinBox()
        self.titresim_input.setRange(0.1, 5.0)
        self.titresim_input.setValue(2.5)
        self.titresim_input.setDecimals(3)
        main_layout.addRow("📳 Titreşim:", self.titresim_input)
        
        # Tork
        self.tork_input = QtWidgets.QDoubleSpinBox()
        self.tork_input.setRange(10.0, 100.0)
        self.tork_input.setValue(55.0)
        main_layout.addRow("⚙️ Tork:", self.tork_input)
        
        main_group.setLayout(main_layout)
        layout.addWidget(main_group)
        
        # Ek sensörler
        extra_group = QtWidgets.QGroupBox("📊 Ek Sensörler")
        extra_layout = QtWidgets.QFormLayout()
        
        # Sensör değerleri ve aralıkları
        sensor_configs = [
            ("sensor_11", "Sensör 11", 40.0, 60.0, 47.5, 1),
            ("sensor_12", "Sensör 12", 500.0, 600.0, 521.0, 0),
            ("sensor_4", "Sensör 4", 1300.0, 1600.0, 1400.0, 0),
            ("sensor_7", "Sensör 7", 500.0, 600.0, 553.0, 0),
            ("sensor_15", "Sensör 15", 5.0, 15.0, 8.4, 1),
            ("sensor_9", "Sensör 9", 8000.0, 10000.0, 9050.0, 0),
            ("sensor_21", "Sensör 21", 20.0, 30.0, 23.3, 1),
            ("sensor_20", "Sensör 20", 30.0, 50.0, 38.9, 1),
            ("sensor_2", "Sensör 2", 600.0, 700.0, 642.0, 0),
            ("sensor_3", "Sensör 3", 1500.0, 1700.0, 1585.0, 0)
        ]
        
        for key, label, min_val, max_val, default, decimals in sensor_configs:
            spinbox = QtWidgets.QDoubleSpinBox()
            spinbox.setRange(min_val, max_val)
            spinbox.setValue(default)
            spinbox.setDecimals(decimals)
            self.sensor_inputs[key] = spinbox
            extra_layout.addRow(f"📈 {label}:", spinbox)
        
        extra_group.setLayout(extra_layout)
        layout.addWidget(extra_group)
        
        group.setLayout(layout)
        return group

    def create_action_buttons(self):
        """Ana işlem butonları"""
        group = QtWidgets.QGroupBox("🚀 İşlemler")
        layout = QtWidgets.QVBoxLayout()
        
        # Tahmin butonu
        self.predict_btn = QtWidgets.QPushButton("🔮 Tahmin Yap")
        self.predict_btn.setStyleSheet("""
            background-color: #3498db; 
            color: white; 
            font-size: 16px; 
            padding: 12px;
            border-radius: 8px;
        """)
        self.predict_btn.clicked.connect(self.make_prediction)
        layout.addWidget(self.predict_btn)
        
        # Analiz butonları
        analysis_layout = QtWidgets.QHBoxLayout()
        
        self.lime_btn = QtWidgets.QPushButton("🔍 LIME")
        self.lime_btn.setStyleSheet("background-color: #9b59b6; color: white;")
        self.lime_btn.clicked.connect(self.show_lime_explanation)
        
        self.shap_btn = QtWidgets.QPushButton("📊 SHAP")
        self.shap_btn.setStyleSheet("background-color: #34495e; color: white;")
        self.shap_btn.clicked.connect(self.show_shap_summary)
        
        analysis_layout.addWidget(self.lime_btn)
        analysis_layout.addWidget(self.shap_btn)
        layout.addLayout(analysis_layout)
        
        # Rapor butonu
        self.report_btn = QtWidgets.QPushButton("📊 Günlük Rapor")
        self.report_btn.setStyleSheet("background-color: #f39c12; color: white;")
        self.report_btn.clicked.connect(self.download_daily_report)
        layout.addWidget(self.report_btn)
        
        group.setLayout(layout)
        return group

    def create_result_display(self):
        """Sonuç gösterim alanı"""
        group = QtWidgets.QGroupBox("📊 Tahmin Sonuçları")
        layout = QtWidgets.QVBoxLayout()
        
        # RUL değeri
        self.rul_display = QtWidgets.QLabel("RUL: --")
        self.rul_display.setFont(QtGui.QFont("Arial", 24, QtGui.QFont.Bold))
        self.rul_display.setAlignment(QtCore.Qt.AlignCenter)
        self.rul_display.setStyleSheet("""
            background-color: #ecf0f1; 
            border: 2px solid #bdc3c7; 
            border-radius: 10px; 
            padding: 20px;
            color: #2c3e50;
        """)
        layout.addWidget(self.rul_display)
        
        # Durum
        self.status_display = QtWidgets.QLabel("Durum: Bekliyor")
        self.status_display.setFont(QtGui.QFont("Arial", 14, QtGui.QFont.Bold))
        self.status_display.setAlignment(QtCore.Qt.AlignCenter)
        self.status_display.setStyleSheet("padding: 10px; color: #7f8c8d;")
        layout.addWidget(self.status_display)
        
        # Mesaj
        self.message_display = QtWidgets.QLabel("Tahmin yapmak için sensör değerlerini girin")
        self.message_display.setWordWrap(True)
        self.message_display.setAlignment(QtCore.Qt.AlignCenter)
        self.message_display.setStyleSheet("padding: 10px; color: #95a5a6; font-style: italic;")
        layout.addWidget(self.message_display)
        
        group.setLayout(layout)
        return group

    def create_right_panel(self):
        """Sağ panel (grafikler ve analiz)"""
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        
        # Grafik alanı
        chart_group = QtWidgets.QGroupBox("📈 Gerçek Zamanlı İzleme")
        chart_layout = QtWidgets.QVBoxLayout()
        
        # Matplotlib figure
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        chart_layout.addWidget(self.canvas)
        
        chart_group.setLayout(chart_layout)
        right_layout.addWidget(chart_group)
        
        # Geçmiş tahminler tablosu
        history_group = QtWidgets.QGroupBox("📋 Tahmin Geçmişi")
        history_layout = QtWidgets.QVBoxLayout()
        
        self.history_table = QtWidgets.QTableWidget()
        self.history_table.setColumnCount(4)
        self.history_table.setHorizontalHeaderLabels(["Zaman", "RUL", "Durum", "Sıcaklık"])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        history_layout.addWidget(self.history_table)
        
        history_group.setLayout(history_layout)
        right_layout.addWidget(history_group)
        
        # İlk grafiği çiz
        self.update_charts()
        
        return right_widget

    def update_charts(self):
        """Grafikleri güncelle"""
        self.figure.clear()
        
        if len(self.prediction_history) > 0:
            ax1 = self.figure.add_subplot(211)
            times = [p['time'] for p in self.prediction_history[-20:]]  # Son 20 tahmin
            ruls = [p['rul'] for p in self.prediction_history[-20:]]
            
            ax1.plot(times, ruls, 'b-o', linewidth=2, markersize=4)
            ax1.set_title('RUL Trendi (Son 20 Tahmin)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('RUL (döngü)')
            ax1.grid(True, alpha=0.3)
            
            # Sıcaklık trendi
            ax2 = self.figure.add_subplot(212)
            temps = [p['sicaklik'] for p in self.prediction_history[-20:]]
            ax2.plot(times, temps, 'r-o', linewidth=2, markersize=4)
            ax2.set_title('Sıcaklık Trendi', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Sıcaklık (°C)')
            ax2.set_xlabel('Zaman')
            ax2.grid(True, alpha=0.3)
        else:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'Henüz tahmin yapılmadı\nTahmin yapmak için sol paneli kullanın', 
                   ha='center', va='center', fontsize=14, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
        
        self.figure.tight_layout()
        self.canvas.draw()

    def start_stream(self):
        """Canlı akış başlat"""
        if not os.path.exists(STREAM_CSV):
            QtWidgets.QMessageBox.warning(self, "Dosya Bulunamadı", 
                                        f"Akış dosyası bulunamadı: {STREAM_CSV}\n\nÖnce sim_stream.py ile veri üretmeyi başlatın.")
            return

        self.is_streaming = True
        self.stream_timer.start(2000)  # 2 saniyede bir
        
        self.start_stream_btn.setEnabled(False)
        self.stop_stream_btn.setEnabled(True)
        
        self.stream_status.setText("🟢 Akış Durumu: Aktif")
        self.stream_status.setStyleSheet("color: #27ae60; font-weight: bold;")
        self.statusBar().showMessage("🔴 Canlı veri akışı başlatıldı")

    def stop_stream(self):
        """Canlı akış durdur"""
        self.is_streaming = False
        self.stream_timer.stop()
        
        self.start_stream_btn.setEnabled(True)
        self.stop_stream_btn.setEnabled(False)
        
        self.stream_status.setText("⚪ Akış Durumu: Durduruldu")
        self.stream_status.setStyleSheet("color: #95a5a6; font-weight: bold;")
        self.statusBar().showMessage("⏹️ Canlı akış durduruldu")

    def update_from_stream(self):
        """Stream dosyasından son satırı oku ve formu güncelle"""
        try:
            if not os.path.exists(STREAM_CSV):
                self.stop_stream()
                return

            df = pd.read_csv(STREAM_CSV)
            
            if len(df) == 0:
                return
                
            # Yeni satır var mı kontrol et
            if len(df) > self.last_row_count:
                self.last_row_count = len(df)
                
                # Son satırı al
                last_row = df.iloc[-1]
                
                # Ana sensör alanlarını güncelle
                self.sicaklik_input.setValue(round(last_row['sicaklik'], 2))
                self.titresim_input.setValue(round(last_row['titresim'], 3))
                self.tork_input.setValue(round(last_row['tork'], 2))
                
                # Otomatik tahmin yap (opsiyonel)
                if hasattr(self, 'auto_predict') and self.auto_predict:
                    self.make_prediction()
                
                # Durum güncelle
                self.statusBar().showMessage(f"🔄 Son güncelleme: {last_row['timestamp']}")

        except Exception as e:
            self.statusBar().showMessage(f"⚠️ Akış hatası: {str(e)}")

    def make_prediction(self):
        """Gerçek model ile tahmin yap"""
        try:
            # Ana sensör değerlerini al
            sicaklik = self.sicaklik_input.value()
            titresim = self.titresim_input.value()
            tork = self.tork_input.value()

            # Tüm sensör verilerini topla
            sensor_data = {
                'sensor_measurement_11': [self.sensor_inputs['sensor_11'].value()],
                'sensor_measurement_12': [self.sensor_inputs['sensor_12'].value() + sicaklik/10],
                'sensor_measurement_4': [self.sensor_inputs['sensor_4'].value() + sicaklik],
                'sensor_measurement_7': [self.sensor_inputs['sensor_7'].value()],
                'sensor_measurement_15': [self.sensor_inputs['sensor_15'].value() + titresim],
                'sensor_measurement_9': [self.sensor_inputs['sensor_9'].value() + tork*10],
                'sensor_measurement_21': [self.sensor_inputs['sensor_21'].value()],
                'sensor_measurement_20': [self.sensor_inputs['sensor_20'].value()],
                'sensor_measurement_2': [self.sensor_inputs['sensor_2'].value() + sicaklik/5],
                'sensor_measurement_3': [self.sensor_inputs['sensor_3'].value() + sicaklik*2]
            }
            
            # DataFrame oluştur
            sensor_df = pd.DataFrame(sensor_data)
            
            # Gerçek model ile tahmin yap
            try:
                # Veriyi ölçekle
                X_scaled = self.scaler.transform(sensor_df[self.features])
                
                # Model ile tahmin
                rul = self.model.predict(X_scaled)[0]
                
                self.statusBar().showMessage("✅ Gerçek ML modeli kullanıldı")
                
            except Exception as model_error:
                # Model hatası durumunda basit hesaplama
                rul = max(10, 200 - (sicaklik/10) - (titresim*20) - (tork/2))
                self.statusBar().showMessage(f"⚠️ Basit hesaplama kullanıldı: {str(model_error)}")
            
            # Bakım kararı al
            maintenance_info = maintenance_decision(rul)
            status = maintenance_info["status"]
            message = maintenance_info["message"]
            color = maintenance_info["color"]

            # Sonuçları güncelle
            self.rul_display.setText(f"RUL: {rul:.1f} döngü")
            self.status_display.setText(f"Durum: {status}")
            self.message_display.setText(message)
            
            # Renkleri güncelle
            status_colors = {
                'CRITICAL': '#e74c3c',
                'PLANNED': '#f39c12', 
                'NORMAL': '#27ae60',
                'UNKNOWN': '#95a5a6'
            }
            
            color = status_colors.get(status, '#95a5a6')
            self.rul_display.setStyleSheet(f"""
                background-color: {color}; 
                color: white;
                border: 2px solid {color}; 
                border-radius: 10px; 
                padding: 20px;
            """)
            
            # Tahmin geçmişine ekle
            prediction_data = {
                'time': datetime.datetime.now().strftime('%H:%M:%S'),
                'rul': rul,
                'status': status,
                'sicaklik': sicaklik,
                'titresim': titresim,
                'tork': tork
            }
            self.prediction_history.append(prediction_data)
            
            # Tabloyu güncelle
            self.update_history_table()
            
            # Grafikleri güncelle
            self.update_charts()

            # CSV'ye logla
            log_data = {
                ColumnNames.TIMESTAMP: datetime.datetime.now().isoformat(),
                ColumnNames.SICAKLIK: sicaklik,
                ColumnNames.TITRESIM: titresim,
                ColumnNames.TORK: tork,
                ColumnNames.RUL: rul,
                ColumnNames.STATUS: status
            }
            
            append_prediction_log(log_data)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Tahmin Hatası", f"Tahmin yapılırken hata oluştu:\n{str(e)}")
            self.statusBar().showMessage(f"❌ Hata: {str(e)}")

    def update_history_table(self):
        """Tahmin geçmişi tablosunu güncelle"""
        self.history_table.setRowCount(len(self.prediction_history))
        
        for i, prediction in enumerate(reversed(self.prediction_history[-10:])):  # Son 10 tahmin
            self.history_table.setItem(i, 0, QtWidgets.QTableWidgetItem(prediction['time']))
            self.history_table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{prediction['rul']:.1f}"))
            self.history_table.setItem(i, 2, QtWidgets.QTableWidgetItem(prediction['status']))
            self.history_table.setItem(i, 3, QtWidgets.QTableWidgetItem(f"{prediction['sicaklik']:.1f}°C"))

    def download_daily_report(self):
        """Günlük rapor indir"""
        try:
            excel_file = daily_report_to_excel()
            
            QtWidgets.QMessageBox.information(self, "Rapor Oluşturuldu", 
                                            f"Günlük Excel raporu başarıyla oluşturuldu!\n\nDosya: {excel_file}")
            
            self.status_label.setText(f"📊 Rapor oluşturuldu: {excel_file}")

        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Rapor Hatası", 
                                        f"Günlük rapor oluşturulamadı:\n{str(e)}")

    def show_lime_explanation(self):
        """LIME açıklaması oluştur ve göster"""
        try:
            # Form değerlerini kontrol et
            sicaklik = self.sicaklik_input.text().strip()
            titresim = self.titresim_input.text().strip()
            tork = self.tork_input.text().strip()

            if not sicaklik or not titresim or not tork:
                QtWidgets.QMessageBox.warning(self, Messages.MISSING_DATA, 
                                            "LIME açıklaması için önce sensör değerlerini girin ve tahmin yapın.")
                return

            # Form değerlerini float'a çevir
            sicaklik_val = float(sicaklik)
            titresim_val = float(titresim)
            tork_val = float(tork)

            # Örnek veri oluştur (gerçek sensör değerleri)
            # Bu basit örnekte sadece 3 değer var, gerçekte tüm sensör değerleri olmalı
            sample_data = {
                'sensor_measurement_11': [47.5],  # FAR
                'sensor_measurement_12': [521.0 + sicaklik_val/10],  # Bypass ratio
                'sensor_measurement_4': [1400.0 + sicaklik_val],  # LPT outlet temp
                'sensor_measurement_7': [553.0],  # HPC outlet pressure
                'sensor_measurement_15': [8.4 + titresim_val],  # Desired core speed
                'sensor_measurement_9': [9050.0 + tork_val*10],  # Physical core speed
                'sensor_measurement_21': [23.3],  # HPC inlet pressure
                'sensor_measurement_20': [38.9],  # HPC inlet temp
                'sensor_measurement_2': [642.0 + sicaklik_val/5],  # LPC outlet temp
                'sensor_measurement_3': [1585.0 + sicaklik_val*2]  # HPC outlet temp
            }
            
            sample_df = pd.DataFrame(sample_data)
            
            # LIME açıklaması oluştur
            self.status_label.setText("🔍 LIME açıklaması oluşturuluyor...")
            
            html_file = explain_instance(
                self.model, 
                self.scaler, 
                sample_df, 
                self.features,
                "reports/lime_explanation.html"
            )
            
            # HTML'i tarayıcıda aç
            open_html_in_browser(html_file)
            
            # Başarı mesajı
            QtWidgets.QMessageBox.information(self, "LIME Açıklaması", 
                                            f"LIME açıklaması başarıyla oluşturuldu!\n\nHTML dosyası tarayıcıda açıldı:\n{html_file}")
            
            self.status_label.setText("✅ LIME açıklaması oluşturuldu")

        except ValueError:
            QtWidgets.QMessageBox.warning(self, Messages.INVALID_DATA, Messages.ENTER_VALID_NUMBERS)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "LIME Hatası", 
                                         f"LIME açıklaması oluşturulamadı:\n{str(e)}")
            self.status_label.setText("❌ LIME hatası")

    def show_shap_summary(self):
        """SHAP Özet (Global) analizi oluştur"""
        try:
            self.status_label.setText("🔍 SHAP özet analizi başlıyor...")
            
            # Örnek veri yükle (300 satır)
            x_sample = load_sample_data(sample_size=300)
            
            # Veriyi ölçekle
            x_scaled = pd.DataFrame(
                self.scaler.transform(x_sample), 
                columns=x_sample.columns,
                index=x_sample.index
            )
            
            # SHAP summary oluştur
            summary_path = shap_summary_png(self.model, x_scaled)
            
            # Dosyayı sistem görüntüleyicisiyle aç
            self.open_image_file(summary_path)
            
            # Başarı mesajı
            QtWidgets.QMessageBox.information(self, "SHAP Özet Analizi", 
                                            f"SHAP özet analizi başarıyla oluşturuldu!\n\nDosya: {summary_path}")
            
            self.status_label.setText("✅ SHAP özet analizi tamamlandı")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "SHAP Özet Hatası", 
                                         f"SHAP özet analizi oluşturulamadı:\n{str(e)}")
            self.status_label.setText("❌ SHAP özet hatası")

    def show_shap_local(self):
        """SHAP Lokal (Tek Örnek) analizi oluştur"""
        try:
            # Form değerlerini kontrol et
            sicaklik = self.sicaklik_input.text().strip()
            titresim = self.titresim_input.text().strip()
            tork = self.tork_input.text().strip()

            if not sicaklik or not titresim or not tork:
                QtWidgets.QMessageBox.warning(self, Messages.MISSING_DATA, 
                                            "SHAP lokal analizi için önce sensör değerlerini girin.")
                return

            self.status_label.setText("🔍 SHAP lokal analizi başlıyor...")

            # Form değerlerini float'a çevir
            sicaklik_val = float(sicaklik)
            titresim_val = float(titresim)
            tork_val = float(tork)

            # Örnek veri oluştur (tüm sensör değerleri)
            sample_data = {
                'sensor_measurement_11': [47.5],  # FAR
                'sensor_measurement_12': [521.0 + sicaklik_val/10],  # Bypass ratio
                'sensor_measurement_4': [1400.0 + sicaklik_val],  # LPT outlet temp
                'sensor_measurement_7': [553.0],  # HPC outlet pressure
                'sensor_measurement_15': [8.4 + titresim_val],  # Desired core speed
                'sensor_measurement_9': [9050.0 + tork_val*10],  # Physical core speed
                'sensor_measurement_21': [23.3],  # HPC inlet pressure
                'sensor_measurement_20': [38.9],  # HPC inlet temp
                'sensor_measurement_2': [642.0 + sicaklik_val/5],  # LPC outlet temp
                'sensor_measurement_3': [1585.0 + sicaklik_val*2]  # HPC outlet temp
            }
            
            sample_df = pd.DataFrame(sample_data)
            
            # Veriyi ölçekle
            x_scaled = pd.DataFrame(
                self.scaler.transform(sample_df), 
                columns=sample_df.columns
            )
            
            # SHAP lokal oluştur
            local_path = shap_local_png(self.model, x_scaled)
            
            # Dosyayı sistem görüntüleyicisiyle aç
            self.open_image_file(local_path)
            
            # Başarı mesajı
            QtWidgets.QMessageBox.information(self, "SHAP Lokal Analizi", 
                                            f"SHAP lokal analizi başarıyla oluşturuldu!\n\nDosya: {local_path}")
            
            self.status_label.setText("✅ SHAP lokal analizi tamamlandı")

        except ValueError:
            QtWidgets.QMessageBox.warning(self, Messages.INVALID_DATA, Messages.ENTER_VALID_NUMBERS)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "SHAP Lokal Hatası", 
                                         f"SHAP lokal analizi oluşturulamadı:\n{str(e)}")
            self.status_label.setText("❌ SHAP lokal hatası")

    def open_image_file(self, image_path):
        """Görsel dosyasını sistem görüntüleyicisiyle aç"""
        try:
            import sys
            if sys.platform.startswith("darwin"):  # macOS
                subprocess.run(["open", image_path])
            elif sys.platform.startswith("win"):  # Windows
                os.startfile(image_path)
            elif sys.platform.startswith("linux"):  # Linux
                subprocess.run(["xdg-open", image_path])
            else:
                print(f"Görsel dosyası: {image_path}")
                
        except Exception as e:
            print(f"⚠️ Görsel dosyası açılamadı: {e}")

    def closeEvent(self, event):
        """Uygulama kapatılırken canlı akışı durdur"""
        if self.is_streaming:
            self.stop_stream()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = PredictionApp()
    window.show()
    sys.exit(app.exec_())