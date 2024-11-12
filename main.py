import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier  # Thay thế ANN bằng MLPClassifier từ scikit-learn
from sklearn.linear_model import LogisticRegression  # Thêm dòng này để import LogisticRegression

# Hàm đọc ảnh và nhãn
def load_data(image_dir):
    images = []
    labels = []
    for label in os.listdir(image_dir):
        label_path = os.path.join(image_dir, label)
        if os.path.isdir(label_path):
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Chuyển ảnh thành đen trắng
                if img is not None:
                    img = cv2.resize(img, (64, 64))  # Chỉnh lại kích thước về 64x64 thay vì 128x128
                    img = img.astype('float32') / 255.0  # Chuẩn hóa giá trị pixel từ [0, 255] về [0, 1]
                    images.append(img.flatten())  # Chuyển ảnh thành vector một chiều
                    labels.append(label)
    return np.array(images), np.array(labels)

# Đường dẫn đến thư mục chứa ảnh
image_dir = 'D:/XLATHGIMAYTINH/bth8/animals'  # Đảm bảo đường dẫn này chính xác

# Tải dữ liệu
X, y = load_data(image_dir)

# Kiểm tra số lượng ảnh
print(f"Loaded {len(X)} images with {len(np.unique(y))} unique labels")

# Kiểm tra ảnh đầu vào
print(f"First 5 labels: {y[:5]}")
print(f"Shape of first image: {X[0].shape}")

# Chuẩn bị dữ liệu huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Chuyển nhãn thành dạng số (Label Encoding)
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# ========================== Logistic Regression ==========================
# Khởi tạo mô hình Logistic Regression với max_iter lớn hơn và tol nhỏ hơn
log_reg = LogisticRegression(max_iter=1000, tol=1e-4)

# Huấn luyện mô hình
log_reg.fit(X_train, y_train)

# Dự đoán
y_pred_log_reg = log_reg.predict(X_test)

# Đánh giá mô hình Logistic Regression
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
print("Accuracy of Logistic Regression: ", accuracy_log_reg)

# ========================== KNN ==========================
# Khởi tạo mô hình KNN với tham số n_neighbors mặc định
knn = KNeighborsClassifier(n_neighbors=3)

# Huấn luyện mô hình
knn.fit(X_train, y_train)

# Dự đoán
y_pred_knn = knn.predict(X_test)

# Đánh giá mô hình KNN
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("Accuracy of KNN: ", accuracy_knn)

# ========================== SVM ==========================
# Khởi tạo mô hình SVM với kernel 'linear'
svm = SVC(kernel='linear')

# Huấn luyện mô hình
svm.fit(X_train, y_train)

# Dự đoán
y_pred_svm = svm.predict(X_test)

# Đánh giá mô hình SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("Accuracy of SVM: ", accuracy_svm)
