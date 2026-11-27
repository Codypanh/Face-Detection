import cv2
import numpy as np
import tensorflow as tf

CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 1. Load model SE-CNN đã huấn luyện
model = tf.keras.models.load_model("best_se_cnn_fer2013.keras")

# 2. Khởi tạo CLAHE & Bộ nhận diện khuôn mặt
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# 3. Mở webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

print("Camera đang mở. Nhấn phím 'q' trên cửa sổ hiển thị để thoát.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển ảnh xám để tìm khuôn mặt
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(40, 40)
    )

    for (x, y, w, h) in faces:
        # 1. Cắt ROI khuôn mặt
        face_roi = gray[y:y+h, x:x+w]
        if face_roi.size == 0:
            continue

        # 2. Tiền xử lý khớp với mô hình SE-CNN
        face_48 = cv2.resize(face_roi, (48, 48))          # Kích thước dùng cho CLAHE
        face_clahe = clahe.apply(face_48)                 # Cân bằng sáng bằng CLAHE
        face_96 = cv2.resize(face_clahe, (96, 96))        # Kích thước model yêu cầu
        face_norm = face_96.astype(np.float32) / 255.0    # Chuẩn hóa về dải [0, 1]

        # Định hình lại tensor: (1, 96, 96, 1)
        face_tensor = np.expand_dims(face_norm, axis=(0, -1))
        face_tensor = np.ascontiguousarray(face_tensor)

        # 3. Dự đoán cảm xúc
        preds = model.predict(face_tensor, verbose=0)[0]
        label = CLASS_NAMES[np.argmax(preds)]
        confidence = np.max(preds) * 100

        # 4. Hiển thị kết quả lên khung hình
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame,
            f"{label}: {confidence:.1f}%",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow("FER-2013 Live Emotion Recognition", frame)

    # Thoát bằng phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()