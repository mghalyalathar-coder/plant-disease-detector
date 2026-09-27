"""
app.py
------
Aplikasi web Flask untuk deteksi penyakit daun tanaman.
User upload gambar daun -> model memprediksi jenis penyakitnya.

Jalankan:
    python app.py
Lalu buka http://127.0.0.1:5000 di browser.
"""

import os
import json
import numpy as np
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ----------------------------
# KONFIGURASI
# ----------------------------
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MODEL_PATH = "model/plant_disease_model.h5"
LABELS_PATH = "model/labels.json"
IMG_SIZE = (224, 224)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----------------------------
# LOAD MODEL & LABEL (sekali saat server start)
# ----------------------------
model = None
idx_to_class = {}

def load_artifacts():
    global model, idx_to_class
    if not os.path.exists(MODEL_PATH):
        print(f"[PERINGATAN] Model belum ditemukan di '{MODEL_PATH}'. "
              f"Jalankan train_model.py terlebih dahulu.")
        return
    model = load_model(MODEL_PATH)
    with open(LABELS_PATH, "r") as f:
        raw = json.load(f)
        idx_to_class = {int(k): v for k, v in raw.items()}
    print("Model & label berhasil dimuat.")

load_artifacts()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def format_label(raw_label):
    """Ubah 'Tomato___Early_blight' menjadi 'Tomato - Early blight'."""
    parts = raw_label.replace("___", "|").replace("_", " ").split("|")
    return " - ".join(p.strip() for p in parts)


def predict_image(img_path):
    img = keras_image.load_img(img_path, target_size=IMG_SIZE)
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    preds = model.predict(arr)[0]
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx]) * 100

    label = idx_to_class.get(top_idx, "Tidak diketahui")

    # Ambil top-3 prediksi untuk ditampilkan juga
    top3_idx = preds.argsort()[-3:][::-1]
    top3 = [
        {"label": format_label(idx_to_class.get(i, "?")), "confidence": round(float(preds[i]) * 100, 2)}
        for i in top3_idx
    ]

    return format_label(label), round(confidence, 2), top3


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template(
            "index.html",
            result=None,
            error="Model belum tersedia. Jalankan 'python train_model.py' dulu untuk melatih model.",
        )

    if "file" not in request.files:
        return render_template("index.html", result=None, error="Tidak ada file yang diupload.")

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", result=None, error="Silakan pilih gambar terlebih dahulu.")

    if not allowed_file(file.filename):
        return render_template("index.html", result=None, error="Format file harus .jpg, .jpeg, atau .png")

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    label, confidence, top3 = predict_image(filepath)

    result = {
        "label": label,
        "confidence": confidence,
        "top3": top3,
        "image_url": url_for("static", filename=f"uploads/{filename}"),
    }
    return render_template("index.html", result=result, error=None)


if __name__ == "__main__":
    app.run(debug=True)
