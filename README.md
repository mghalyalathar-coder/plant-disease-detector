# 🌿 Deteksi Penyakit Daun Tanaman

Proyek web sederhana: upload foto daun tanaman → model AI memprediksi jenis penyakitnya.
Stack: **Flask** (backend) + **TensorFlow/Keras** (model, transfer learning MobileNetV2) + dataset **PlantVillage**.

## Struktur Folder

```
plant-disease-detector/
├── app.py                  # Aplikasi Flask (web server)
├── train_model.py          # Script training model
├── requirements.txt        # Daftar library python
├── dataset/
│   └── PlantVillage/       # Taruh dataset dari Kaggle di sini
├── model/                  # Model hasil training akan tersimpan di sini
├── static/
│   ├── style.css
│   └── uploads/            # Folder gambar yang diupload user
└── templates/
    └── index.html          # Halaman web
```

## Langkah 1 — Buka di VSCode

1. Extract/salin folder `plant-disease-detector` ke komputer kamu.
2. Buka VSCode → `File > Open Folder` → pilih folder `plant-disease-detector`.

## Langkah 2 — Buat Virtual Environment

Buka terminal di VSCode (`Ctrl+`\`\`):

```bash
python -m venv venv
```

Aktifkan:
- Windows: `venv\Scripts\activate`
- Mac/Linux: `source venv/bin/activate`

## Langkah 3 — Install Library

```bash
pip install -r requirements.txt
```

## Langkah 4 — Download Dataset PlantVillage dari Kaggle

1. Buat akun di [kaggle.com](https://www.kaggle.com) kalau belum punya.
2. Cari dataset **"PlantVillage"** (contoh: `emmarex/plantdisease` atau `abdallahalidev/plantvillage-dataset`).
3. Download file zip, lalu extract.
4. Susun ulang foldernya sehingga strukturnya seperti ini (setiap subfolder = 1 kelas penyakit):

```
dataset/PlantVillage/
    Tomato___Early_blight/
        img1.jpg
        img2.jpg
    Tomato___healthy/
        img1.jpg
    Potato___Late_blight/
        ...
```

> Kalau hasil download strukturnya beda (misal ada folder `train/` `val/` terpisah), gabungkan saja semua gambar per kelas ke satu folder seperti contoh di atas — `train_model.py` yang akan membagi otomatis 80% training / 20% validasi.

## Langkah 5 — Training Model

```bash
python train_model.py
```

Proses ini akan:
- Membaca semua gambar dari `dataset/PlantVillage`
- Melatih model MobileNetV2 (transfer learning) selama beberapa epoch
- Menyimpan model terbaik ke `model/plant_disease_model.h5`
- Menyimpan daftar label ke `model/labels.json`
- Menyimpan grafik akurasi/loss ke `model/training_history.png`

⏱️ Catatan: training bisa memakan waktu cukup lama tanpa GPU (bisa 30–90 menit tergantung jumlah data). Kalau punya GPU NVIDIA, install `tensorflow` versi GPU agar lebih cepat.

Kalau ingin coba cepat dulu sebelum pakai dataset penuh, kamu bisa pakai subset kecil (misal 3-5 kelas, 100 gambar per kelas) supaya training hanya beberapa menit.

## Langkah 6 — Jalankan Web App

```bash
python app.py
```

Buka browser ke: **http://127.0.0.1:5000**

Upload foto daun → klik "Deteksi Sekarang" → hasil prediksi + persentase keyakinan akan muncul.

## Kustomisasi Lanjutan (opsional)

- **Ganti arsitektur model**: di `train_model.py`, ganti `MobileNetV2` dengan `EfficientNetB0` atau `ResNet50` dari `tensorflow.keras.applications`.
- **Fine-tuning**: setelah training awal, buka freeze beberapa layer terakhir base_model (`base_model.trainable = True` untuk sebagian layer) dan lanjutkan training dengan learning rate lebih kecil, agar akurasi naik lagi.
- **Deploy online**: bisa deploy ke Render, Railway, atau Hugging Face Spaces. File model `.h5` biasanya berukuran puluhan MB, pastikan platform hosting mendukung ukuran itu.
- **Ganti Flask jadi FastAPI**: kalau mau versi API murni (tanpa HTML), logika `predict_image()` di `app.py` bisa dipindah ke endpoint FastAPI dengan `UploadFile`.

## Troubleshooting

| Masalah | Solusi |
|---|---|
| `ModuleNotFoundError: No module named 'tensorflow'` | Pastikan virtual environment aktif, lalu `pip install -r requirements.txt` |
| Training sangat lambat | Kurangi jumlah data / epoch dulu untuk uji coba, atau pakai Google Colab (ada GPU gratis) untuk training, lalu download file `.h5`-nya ke folder `model/` |
| `FileNotFoundError: dataset/PlantVillage` | Pastikan struktur folder dataset sudah sesuai Langkah 4 |
| Model prediksi ngawur | Tambah epoch, tambah variasi data augmentasi, atau cek apakah dataset per kelas jumlahnya seimbang |
