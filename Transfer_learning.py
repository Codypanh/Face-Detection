import os
import cv2
import kagglehub
import numpy as np
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from keras import layers
from keras.applications import MobileNetV2
from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# ==========================================
# 0. DOWNLOAD DATASET
# ==========================================

dataset_path = kagglehub.dataset_download(
    "msambare/fer2013"
)

train_dir = os.path.join(
    dataset_path,
    "train"
)

test_dir = os.path.join(
    dataset_path,
    "test"
)

print("Dataset path:", dataset_path)
print("Train path:", train_dir)
print("Test path:", test_dir)

IMG_SIZE = (96, 96)

ORIGINAL_SIZE = (48, 48)

BATCH_SIZE = 64

NUM_CLASSES = 7

SEED = 42

AUTOTUNE = tf.data.AUTOTUNE

# ==========================================
# 1. LOAD DATASET (48x48x1)
# ==========================================
train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=ORIGINAL_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED
)

test_ds_raw = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=ORIGINAL_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

class_names = train_ds_raw.class_names
disgust_index = class_names.index("disgust") if "disgust" in class_names else 1

# ==========================================
# 2. TACH TRAIN THANH TRAIN + VALIDATION
# ==========================================
def clean_image_data(images, labels, min_var=100.0, min_mean=15.0, max_mean=240.0):
    valid_indices = []
    
    for idx, img in enumerate(images):
        img_flat = img.astype(np.float32)
        variance = np.var(img_flat)
        mean_val = np.mean(img_flat)
        
        # Loại bỏ ảnh phẳng (low-variance) và ảnh quá tối / quá chói
        if (variance >= min_var) and (min_mean <= mean_val <= max_mean):
            valid_indices.append(idx)
            
    print(f"--> Đã loại bỏ {len(images) - len(valid_indices)} ảnh nhiễu/lỗi chất lượng kém.")
    return images[valid_indices], labels[valid_indices]
def dataset_to_numpy(dataset):
    images = []
    labels = []

    for batch_images, batch_labels in dataset:
        images.append(
            batch_images.numpy()
        )
        labels.append(
            batch_labels.numpy()
        )

    images = np.concatenate(
        images,
        axis=0
    )

    labels = np.concatenate(
        labels,
        axis=0
    )

    return images, labels

# 1. Trích xuất toàn bộ dữ liệu Train & Test
X_train_full, y_train_full = dataset_to_numpy(train_ds_raw)
X_test, y_test = dataset_to_numpy(test_ds_raw)

# 2. Làm sạch tập Train
X_train_full, y_train_full = clean_image_data(X_train_full, y_train_full)

# 3. Phân chia Train / Validation (Stratified để bảo toàn tỷ lệ nhãn)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.15,
    random_state=SEED,
    stratify=y_train_full
)

print(f"Tập Train: {X_train.shape[0]} mẫu")
print(f"Tập Validation: {X_val.shape[0]} mẫu")
print(f"Tập Test: {X_test.shape[0]} mẫu")

# ==========================================
# 3. PREPROCESSING (CLAHE & NORMALIZATION)
# ==========================================
def apply_clahe_np(image):
    # CLAHE yêu cầu uint8 [0, 255]
    img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    if img_uint8.ndim == 3 and img_uint8.shape[-1] == 1:
        img_uint8 = img_uint8[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(img_uint8)
    return equalized[:, :, np.newaxis].astype(np.float32)

def preprocess_image(
    image,
    label
):
    image = tf.numpy_function(
        apply_clahe_np,
        [image],
        tf.float32
    )

    image.set_shape(
        [48, 48, 1]
    )

    # Grayscale → RGB
    image = tf.image.grayscale_to_rgb(
        image
    )

    # 48×48 → 96×96
    image = tf.image.resize(
        image,
        IMG_SIZE
    )

    # MobileNetV2 preprocessing
    image = tf.keras.applications.mobilenet_v2.preprocess_input(
        image
    )

    return image, label

# ==========================================
# 4. CONDITIONAL DATA AUGMENTATION
# ==========================================
# Pipeline Augment chuẩn
normal_aug = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.08),
    layers.RandomContrast(0.08),
])

# Pipeline Augment mạnh hơn riêng cho lớp Disgust (chống mất cân bằng dữ liệu)
disgust_aug = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.12),
    layers.RandomZoom(0.12),
    layers.RandomContrast(0.15),
    layers.RandomTranslation(0.1, 0.1),
])

def apply_augmentation(
    image,
    label
):
    def strong_augmentation():
        return disgust_aug(
            image,
            training=True
        )

    def normal_augmentation():
        return normal_aug(
            image,
            training=True
        )

    image = tf.cond(
        tf.equal(
            label,
            disgust_index
        ),
        strong_augmentation,
        normal_augmentation
    )

    return image, label

# Xây dựng TF Data Pipeline tối ưu I/O
AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices(
    (
        X_train,
        y_train
    )
)
train_ds = train_ds.map(
    preprocess_image,
    num_parallel_calls=AUTOTUNE
).map(
    apply_augmentation,
    num_parallel_calls=AUTOTUNE
)
train_ds = train_ds.shuffle(
    4096,
    seed=SEED
)
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(
    AUTOTUNE
)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = (
    val_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE) # <-- Chỉ tiền xử lý
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

# ==========================================
# 3. TEST DS (Chỉ Preprocess, KHÔNG Augment, KHÔNG Shuffle)
# ==========================================
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_ds = (
    test_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE) # <-- Chỉ tiền xử lý
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

# ==========================================
# 5. MODEL ARCHITECTURE (MobileNetV2)
# ==========================================
base_model = MobileNetV2(
    input_shape=(96, 96, 3),
    include_top=False,
    weights="imagenet"
)

# Giai đoạn 1: Đóng băng base model
base_model.trainable = False

inputs = keras.Input(shape=(96, 96, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs, outputs)

# ==========================================
# 6. CLASS WEIGHT
# ==========================================
classes = np.unique(y_train)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(
        classes,
        class_weights_array
    )
}

print("\nClass weights:")

for cls, weight in class_weights.items():
    print(
        f"{class_names[cls]:<10} : {weight:.4f}"
    )

# ==========================================
# 7. FEATURE EXTRACTION TRAINING
# ==========================================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_phase1 = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint("best_feature_extraction(2).keras", monitor="val_loss", save_best_only=True)
]

callbacks_phase2 = [
    EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        "best_mobilenetv2_fer2013.keras",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    )
]

print("--- GIAI ĐOẠN 1: FEATURE EXTRACTION ---")
model.fit(
    train_ds,
    
    validation_data=val_ds,
    epochs=18,
    callbacks=callbacks_phase1
)

# ==========================================
# 7. FINE-TUNING
# ==========================================
base_model.trainable = True
# Đóng băng các tầng đầu, chỉ mở 40 tầng cuối để tránh phá vỡ trọng số ImageNet
for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5), # LR thấp hơn khi fine-tune
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("--- GIAI ĐOẠN 2: FINE-TUNING ---")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,
    callbacks=callbacks_phase2
)

# ==========================================
# 8. EVALUATION
# ==========================================
test_loss, test_acc = model.evaluate(test_ds)
print(f"\nFinal Test Loss: {test_loss:.4f}")
print(f"Final Test Accuracy: {test_acc:.4f}")