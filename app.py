from flask import Flask, render_template, request
import matplotlib
matplotlib.use('Agg')  # Gunakan 'Agg' untuk non-GUI, cocok buat web server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
import base64
import os
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer

app = Flask(__name__)

# Load model dan scaler saat aplikasi dimulai
model_path = 'model_banjir.h5'
scaler_path = 'scaler.save'

# Custom InputLayer untuk menangani batch_shape yang deprecated di Keras baru
# Solusi: Monkey patch InputLayer.from_config untuk menangani batch_shape
_original_from_config = InputLayer.from_config

def _patched_from_config(cls, config):
    # Konversi batch_shape ke input_shape jika ada
    if isinstance(config, dict) and 'batch_shape' in config and config['batch_shape']:
        config = config.copy()  # Buat copy untuk menghindari modifikasi langsung
        config['input_shape'] = config['batch_shape'][1:]
        del config['batch_shape']
    return _original_from_config(config)

# Apply monkey patch
InputLayer.from_config = classmethod(_patched_from_config)

# Cek apakah model dan scaler ada
if os.path.exists(model_path) and os.path.exists(scaler_path):
    try:
        # Load model dengan compile=False (monkey patch sudah menangani batch_shape)
        model = load_model(model_path, compile=False)
        # Compile model setelah dimuat
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        scaler = joblib.load(scaler_path)
        model_loaded = True
        print("✓ Model dan scaler berhasil dimuat!")
    except Exception as e:
        print(f"⚠ Error saat memuat model: {str(e)}")
        import traceback
        traceback.print_exc()
        model = None
        scaler = None
        model_loaded = False
else:
    model = None
    scaler = None
    model_loaded = False
    print("⚠ Model atau scaler tidak ditemukan. Pastikan file model_banjir.h5 dan scaler.save ada.")

@app.route('/')
def index():
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil inputan user
        suhu = float(request.form['suhu'])  # Tavg
        curah_hujan = float(request.form['curah_hujan'])  # RR
        kelembapan = float(request.form['kelembapan'])  # RH_avg

        # Validasi input
        if suhu < -50 or suhu > 50:
            return render_template('index.html', error="Suhu harus antara -50°C hingga 50°C", model_loaded=model_loaded)
        if curah_hujan < 0 or curah_hujan > 1000:
            return render_template('index.html', error="Curah hujan harus antara 0-1000 mm", model_loaded=model_loaded)
        if kelembapan < 0 or kelembapan > 100:
            return render_template('index.html', error="Kelembapan harus antara 0-100%", model_loaded=model_loaded)

        # Prediksi menggunakan model ANN
        if model_loaded:
            # Siapkan data sesuai format model: [Tavg, RH_avg, RR]
            input_data = np.array([[suhu, kelembapan, curah_hujan]])
            
            # Scale data menggunakan scaler yang sama
            input_scaled = scaler.transform(input_data)
            
            # Prediksi
            probabilitas_banjir = model.predict(input_scaled, verbose=0)[0][0]
            hasil = 'Banjir' if probabilitas_banjir > 0.5 else 'Tidak Banjir'
        else:
            # Fallback jika model tidak dimuat
            probabilitas_banjir = (curah_hujan * 0.4 + kelembapan * 0.3 - suhu * 0.2) / 100
            probabilitas_banjir = max(0, min(1, probabilitas_banjir))  # Clamp antara 0-1
            hasil = 'Banjir' if probabilitas_banjir > 0.5 else 'Tidak Banjir'

        # Buat grafik visualisasi
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Grafik 1: Bar chart probabilitas
        colors = ['#28a745' if hasil == 'Tidak Banjir' else '#dc3545']
        ax1.bar(['Probabilitas Banjir'], [probabilitas_banjir * 100], 
                color=colors[0], alpha=0.7, edgecolor='black', linewidth=2)
        ax1.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Threshold (50%)')
        ax1.set_ylim(0, 100)
        ax1.set_ylabel('Probabilitas (%)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Hasil Prediksi: {hasil}', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Tambahkan nilai di atas bar
        ax1.text(0, probabilitas_banjir * 100 + 2, f'{probabilitas_banjir * 100:.2f}%', 
                ha='center', fontsize=14, fontweight='bold')

        # Grafik 2: Input values comparison
        features = ['Suhu\n(Tavg)', 'Kelembapan\n(RH_avg)', 'Curah Hujan\n(RR)']
        values = [suhu, kelembapan, curah_hujan]
        colors_bar = ['#ff6b6b', '#4ecdc4', '#45b7d1']
        
        bars = ax2.bar(features, values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Nilai Input', fontsize=12, fontweight='bold')
        ax2.set_title('Nilai Input yang Dimasukkan', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Tambahkan nilai di atas setiap bar
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()

        # Simpan ke memory buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        grafik_base64 = base64.b64encode(buf.getvalue()).decode('ascii')
        plt.close(fig)

        # Format probabilitas untuk ditampilkan
        probabilitas_persen = probabilitas_banjir * 100

        return render_template('result.html', 
                             hasil=hasil, 
                             grafik=grafik_base64,
                             probabilitas=probabilitas_persen,
                             suhu=suhu,
                             curah_hujan=curah_hujan,
                             kelembapan=kelembapan,
                             model_loaded=model_loaded)

    except ValueError as e:
        return render_template('index.html', error="Mohon masukkan nilai yang valid untuk semua field", model_loaded=model_loaded)
    except Exception as e:
        return render_template('index.html', error=f"Terjadi kesalahan: {str(e)}", model_loaded=model_loaded)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
