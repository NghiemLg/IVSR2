import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from train import SimpleCNN  # Giả sử SimpleCNN được định nghĩa trong train.py
import os

# Khởi tạo mô hình
num_classes = 8  # Số lớp của bạn
model = SimpleCNN(num_classes=num_classes)

# Load toàn bộ mô hình từ file animal_classification_model.pth
model.load_state_dict(torch.load('best_model.pth'))
model.eval()  # Chuyển sang chế độ đánh giá

# Đường dẫn tới thư mục chứa ảnh cần dự đoán
folder_path = r'D:\New_IVSR\data\spider'  # Thay đổi đường dẫn đến thư mục của bạn

# Các transform để tiền xử lý ảnh
test_transforms = transforms.Compose([
    transforms.Resize((128, 128)),  # Thay đổi kích thước ảnh
    transforms.ToTensor(),        # Chuyển đổi ảnh thành tensor và chuẩn hóa
])

# Định nghĩa danh sách các lớp (tên lớp)
class_names = ['butterfly', 'chicken', 'cow', 'dao son', 'horse', 'sheep', 'spider', 'squirrel']

# Duyệt qua tất cả các file trong thư mục
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Kiểm tra định dạng ảnh
        img_path = os.path.join(folder_path, filename)

        # Load ảnh và chuẩn bị
        img = Image.open(img_path)
        img = test_transforms(img)
        img = img.unsqueeze(0)  # Thêm chiều batch

        # Dự đoán
        with torch.no_grad():  # Tắt tính toán gradient
            predictions = model(img)

        # Tìm lớp có xác suất cao nhất
        predicted_class_index = torch.argmax(predictions, dim=1)

        # Lấy nhãn dự đoán
        predicted_label = class_names[predicted_class_index.item()]
        print(f"Ảnh: {filename}, Predicted class: {predicted_label}")

        # Hiển thị ảnh và chú thích (tùy chọn)
        plt.imshow(Image.open(img_path))
        plt.title(f'Predicted: {predicted_label}')
        plt.axis('off')
        plt.show()
