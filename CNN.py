import os
import cv2
import kagglehub
import numpy as np
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras import layers
from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


# 0. TẢI DATASET FER-2013

dataset_path = kagglehub.dataset_download("msambare/fer2013")
train_dir = os.path.join(dataset_path, "train")
test_dir = os.path.join(dataset_path, "test")

ORIGINAL_SIZE = (48, 48)
BATCH_SIZE = 64
NUM_CLASSES = 7
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


# 1. NẠP DỮ LIỆU THÔ (48x48x1)

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

def dataset_to_numpy(dataset):
    images, labels = [], []
    for batch_images, batch_labels in dataset:
        images.append(batch_images.numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(images, axis=0), np.concatenate(labels, axis=0)

X_train_full, y_train_full = dataset_to_numpy(train_ds_raw)
X_test, y_test = dataset_to_numpy(test_ds_raw)


# 2. DATA CLEANING & CHUYỂN ONE-HOT ENCODING

def clean_image_data(images, labels, min_var=100.0, min_mean=15.0, max_mean=240.0):
    valid_indices = []
    for idx, img in enumerate(images):
        img_flat = img.astype(np.float32)
        if (np.var(img_flat) >= min_var) and (min_mean <= np.mean(img_flat) <= max_mean):
            valid_indices.append(idx)
            
    print(f"--> [Clean Data] Đã loại bỏ {len(images) - len(valid_indices)} ảnh rác/kém chất lượng.")
    return images[valid_indices], labels[valid_indices]

X_train_full, y_train_full = clean_image_data(X_train_full, y_train_full)

# Phân chia Train / Validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.15,
    random_state=SEED,
    stratify=y_train_full
)

# Chuyển nhãn sang One-Hot để dùng Label Smoothing
y_train_oh = keras.utils.to_categorical(y_train, NUM_CLASSES).astype(np.float32)
y_val_oh = keras.utils.to_categorical(y_val, NUM_CLASSES).astype(np.float32)
y_test_oh = keras.utils.to_categorical(y_test, NUM_CLASSES).astype(np.float32)

print(f"Số lượng mẫu: Train = {len(X_train)} | Val = {len(X_val)} | Test = {len(X_test)}")


# 3. TIỀN XỬ LÝ (CLAHE + CHUẨN HÓA VỀ [0, 1])

def apply_clahe_np(image):
    img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    if img_uint8.ndim == 3 and img_uint8.shape[-1] == 1:
        img_uint8 = img_uint8[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(img_uint8)
    return equalized[:, :, np.newaxis].astype(np.float32)

def preprocess_image(image, label):
    image = tf.numpy_function(apply_clahe_np, [image], tf.float32)
    image.set_shape([48, 48, 1])
    image = tf.image.resize(image, [96, 96])  # thêm dòng này
    image = image / 255.0
    return image, label



# 4. DATA AUGMENTATION (CHỈ DÙNG KHI TRAIN)
# 4.1: OVERSAMPLE DISGUST TRƯỚC KHI BUILD PIPELINE

disgust_idx = class_names.index("disgust")

disgust_mask = y_train == disgust_idx
X_disgust = X_train[disgust_mask]
y_disgust = y_train[disgust_mask]

# Repeat 4 lần để đưa disgust lên ~12% (từ ~3%)
X_train_bal = np.concatenate([X_train, np.tile(X_disgust, (4, 1, 1, 1))])
y_train_bal = np.concatenate([y_train, np.tile(y_disgust, 4)])

# Shuffle lại sau khi oversample
perm = np.random.permutation(len(X_train_bal))
X_train_bal = X_train_bal[perm]
y_train_bal = y_train_bal[perm]

# Chuyển one-hot
y_train_bal_oh = keras.utils.to_categorical(y_train_bal, NUM_CLASSES)

print(f"Sau oversample: {len(X_train_bal)} mẫu")
for i, name in enumerate(class_names):
    count = np.sum(y_train_bal == i)
    print(f"  {name:<10}: {count} ({count/len(y_train_bal)*100:.1f}%)")



# KHỞI TẠO CÁC LAYER AUGMENTATION BÊN NGOÀI (1 lần duy nhất) Note

aug_rotation    = keras.layers.RandomRotation(
    factor=0.08,
    fill_mode="reflect"
)
aug_translation = keras.layers.RandomTranslation(
    height_factor=0.06,
    width_factor=0.06,
    fill_mode="reflect"
)


# HÀM AUGMENTATION — GỌI layer, không khởi tạo

@tf.function
def augment_batch(images, labels):
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_brightness(images, max_delta=0.15)
    images = tf.image.random_contrast(images, lower=0.8, upper=1.2)
    images = tf.clip_by_value(images, 0.0, 1.0)
    images = aug_rotation(images, training=True)    # gọi layer đã tạo sẵn
    images = aug_translation(images, training=True) # gọi layer đã tạo sẵn
    return images, labels

@tf.function
def mixup_batch(images, labels, alpha=0.2):
    batch_size = tf.shape(images)[0]
    lam = tf.random.uniform([], 0.0, alpha)
    indices = tf.random.shuffle(tf.range(batch_size))
    labels = tf.cast(labels, tf.float32)
    mixed_images = lam * images + (1.0 - lam) * tf.gather(images, indices)
    mixed_labels = lam * labels + (1.0 - lam) * tf.gather(labels, indices)
    return mixed_images, mixed_labels

@tf.function
def augment_with_mixup(images, labels):
    images, labels = augment_batch(images, labels)
    labels = tf.cast(labels, tf.float32)
    apply_mixup = tf.random.uniform([]) > 0.5
    images, labels = tf.cond(
        apply_mixup,
        lambda: mixup_batch(images, labels, alpha=0.2),
        lambda: (images, labels)
    )
    return images, labels

# 4.4: Xây dựng tf.data Pipelines
train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train_bal, y_train_bal_oh))
    .map(preprocess_image, num_parallel_calls=AUTOTUNE)  # CLAHE + normalize
    .shuffle(buffer_size=8192, seed=SEED, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE, drop_remainder=True)   # drop_remainder để batch size cố định cho @tf.function
    .map(augment_with_mixup, num_parallel_calls=AUTOTUNE)  # augment theo batch
    .prefetch(AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val_oh))
    .map(preprocess_image, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)
test_ds = (
    tf.data.Dataset.from_tensor_slices((X_test, y_test_oh))
    .map(preprocess_image, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)


# 5. XÂY DỰNG CUSTOM CNN VỚI SE-BLOCK (SENet)

def se_block(input_tensor, ratio=8):
    """Squeeze-and-Excitation Block"""
    channels = input_tensor.shape[-1]
    # Squeeze: Global Average Pooling
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Reshape((1, 1, channels))(se)
    # Excitation: 2 lớp Dense (Bottleneck)
    se = layers.Dense(max(channels // ratio, 16), activation="relu", use_bias=False)(se)
    se = layers.Dense(channels, activation="sigmoid", use_bias=False)(se)
    # Scale: Tái gán trọng số cho từng channel
    return layers.multiply([input_tensor, se])

def se_residual_block(x, filters, downsample=False, dropout_rate=0.2):
    shortcut = x
    strides = 2 if downsample else 1

    x = layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.Dropout(dropout_rate)(x)   # ✅ dropout GIỮA các conv, không sau add

    x = layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = se_block(x, ratio=16)

    if downsample or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=strides, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation("elu")(x)
    
    return x


def build_se_cnn(input_shape=(96, 96, 1), num_classes=7):
    inputs = keras.Input(shape=input_shape)
    
    # Stem Block: 96x96 → 48x48
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.MaxPooling2D(2)(x)          # 96 → 48

    # Stage 1: 48x48 → 24x24
    x = se_residual_block(x, 64,  downsample=False, dropout_rate=0.2)
    x = layers.MaxPooling2D(2)(x)

    # Stage 2: 24x24 → 12x12
    x = se_residual_block(x, 128, downsample=False, dropout_rate=0.2)
    x = layers.MaxPooling2D(2)(x)

    # Stage 3: 12x12 → 6x6
    x = se_residual_block(x, 256, downsample=False, dropout_rate=0.2)
    x = layers.MaxPooling2D(2)(x)

    # Stage 4: 6x6 (giữ nguyên resolution, không dropout trong block)
    x = se_residual_block(x, 512, downsample=False, dropout_rate=0.0)

    # Classifier Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="elu")(x)
    x = layers.Dropout(0.5)(x)             # dropout tập trung ở head
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    return keras.Model(inputs, outputs, name="Custom_SE_ResNet_v2")

model = build_se_cnn()
model.summary()

# 6. HUẤN LUYỆN VỚI LABEL SMOOTHING
# Sử dụng Label Smoothing = 0.1 để chống overconfidence & chống nhiễu nhãn
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=10,
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
        "best_se_cnn_fer2013.keras",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    )
]

print("\n--- BẮT ĐẦU HUẤN LUYỆN CUSTOM SE-CNN ---")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=callbacks
)


# 7. ĐÁNH GIÁ TRÊN TEST SET & MA TRẬN NHẦM LẪN
print("\n--- ĐÁNH GIÁ TRÊN TEST SET ---")
test_loss, test_acc = model.evaluate(test_ds)
print(f"Final Test Loss: {test_loss:.4f}")
print(f"Final Test Accuracy: {test_acc:.4f}")

# Dự đoán và xuất Classification Report
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)
print("\n--- CLASSIFICATION REPORT (7 EMOTIONS) ---")
print(classification_report(y_test, y_pred, target_names=class_names))