"""
train_model.py
----------------
Melatih model klasifikasi penyakit daun tanaman menggunakan
transfer learning (MobileNetV2) di atas dataset PlantVillage.

Struktur folder dataset yang diharapkan (hasil ekstrak dari Kaggle):

dataset/
    PlantVillage/
        Tomato___Early_blight/
            img1.jpg
            img2.jpg
            ...
        Tomato___healthy/
            ...
        Potato___Late_blight/
            ...
        ... (setiap folder = 1 kelas/label)

Cara pakai:
    python train_model.py
"""

import os
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ----------------------------
# KONFIGURASI
# ----------------------------
DATASET_DIR = "dataset/PlantVillage"   # sesuaikan dengan lokasi dataset kamu
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
MODEL_OUTPUT_DIR = "model"
MODEL_PATH = os.path.join(MODEL_OUTPUT_DIR, "plant_disease_model.h5")
LABELS_PATH = os.path.join(MODEL_OUTPUT_DIR, "labels.json")

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)


def check_dataset():
    if not os.path.isdir(DATASET_DIR):
        raise FileNotFoundError(
            f"Folder dataset '{DATASET_DIR}' tidak ditemukan.\n"
            "Download dataset PlantVillage dari Kaggle, ekstrak, lalu taruh "
            "di dalam folder 'dataset/' sehingga strukturnya menjadi "
            "'dataset/PlantVillage/<nama_kelas>/<gambar>.jpg'."
        )
    classes = [d for d in os.listdir(DATASET_DIR)
               if os.path.isdir(os.path.join(DATASET_DIR, d))]
    if len(classes) == 0:
        raise ValueError("Tidak ada folder kelas di dalam dataset. Cek ulang struktur foldernya.")
    print(f"Ditemukan {len(classes)} kelas: {classes[:5]}{'...' if len(classes) > 5 else ''}")


def build_data_generators():
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2,  # 80% train, 20% validasi
    )

    train_gen = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_gen = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    return train_gen, val_gen


def build_model(num_classes):
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # freeze dulu untuk transfer learning tahap awal

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history.get("val_accuracy", []), label="val")
    axes[0].set_title("Akurasi")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history.get("val_loss", []), label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_OUTPUT_DIR, "training_history.png"))
    print(f"Grafik training disimpan di {MODEL_OUTPUT_DIR}/training_history.png")


def main():
    check_dataset()
    train_gen, val_gen = build_data_generators()

    # Simpan mapping index -> nama kelas, dipakai lagi saat prediksi di app.py
    idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
    with open(LABELS_PATH, "w") as f:
        json.dump(idx_to_class, f, indent=2)
    print(f"Label kelas disimpan di {LABELS_PATH}")

    model = build_model(num_classes=len(train_gen.class_indices))
    model.summary()

    callbacks = [
        EarlyStopping(monitor="loss", patience=3, restore_best_weights=True),
    ]

    # Hanya pakai ModelCheckpoint kalau ada data validasi yang cukup
    if val_gen.samples > 0:
        callbacks.append(
            ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True)
        )

    history = model.fit(
        train_gen,
        validation_data=val_gen if val_gen.samples > 0 else None,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # PENTING: paksa simpan model di akhir, apapun yang terjadi dengan callback di atas
    model.save(MODEL_PATH)

    plot_history(history)
    print(f"\nSelesai! Model terbaik tersimpan di: {MODEL_PATH}")

    # Verifikasi file benar-benar ada
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"Verifikasi: file model ditemukan, ukuran {size_mb:.2f} MB")
    else:
        print("PERINGATAN: file model TIDAK ditemukan setelah disimpan! Ada masalah.")


if __name__ == "__main__":
    main()