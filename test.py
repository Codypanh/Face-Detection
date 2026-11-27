import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Danh sách 7 nhãn cảm xúc theo thứ tự chuẩn của thư mục
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 2. Load model tốt nhất đã lưu
MODEL_PATH = "best_finetuned_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)
print("--> Đã load model thành công!")

# 3. Hàm tiền xử lý đúng chuẩn pipeline lúc train
def preprocess_input_image(image_path):
    # Đọc ảnh xám
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
    
    # Cắt / Resize sơ bộ về 48x48 nếu cần
    img_48 = cv2.resize(img, (48, 48))
    
    # Áp dụng CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_48)
    
    # Chuyển Grayscale -> RGB (3 kênh)
    img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
    
    # Resize lên kích thước mạng MobileNetV2 (96x96)
    img_resized = cv2.resize(img_rgb, (96, 96))
    img_array = img_resized.astype(np.float32)
    
    # Chuẩn hóa về [-1, 1] theo chuẩn MobileNetV2
    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    # Thêm batch dimension: (1, 96, 96, 3)
    img_tensor = np.expand_dims(img_preprocessed, axis=0)
    
    return img_tensor, img

# 4. Chạy dự đoán
image_path = "test2.jpg"  # Thay bằng đường dẫn ảnh khuôn mặt của bạn
input_tensor, original_img = preprocess_input_image(image_path)

preds = model.predict(input_tensor, verbose=0)[0]
pred_class = CLASS_NAMES[np.argmax(preds)]
confidence = np.max(preds) * 100

print(f"\nKết quả dự đoán: {pred_class.upper()} ({confidence:.2f}%)")
print("-" * 35)
for emotion, prob in zip(CLASS_NAMES, preds):
    print(f"{emotion.ljust(10)}: {prob * 100:6.2f}%")

# 5. Hiển thị ảnh và biểu đồ xác suất
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(original_img, cmap='gray')
plt.title(f"Dự đoán: {pred_class} ({confidence:.1f}%)")
plt.axis('off')

plt.subplot(1, 2, 2)
bars = plt.barh(CLASS_NAMES, preds * 100, color='skyblue')
plt.xlabel('Xác suất (%)')
plt.xlim(0, 100)
for bar in bars:
    plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f"{bar.get_width():.1f}%", va='center')

plt.tight_layout()
plt.show()