from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import random_split
import os
from PIL import Image
from torch.utils.data import Dataset
import torch

class CustomDataset:
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_map = {}  # Bản đồ ánh xạ nhãn
        self._load_data(root_dir)
        
        # Tạo một bản đồ ánh xạ nhãn sang số
        unique_labels = sorted(set(self.labels))
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}

    def _load_data(self, current_dir):
        for entry in os.scandir(current_dir):
            if entry.is_dir():
                self._load_data(entry.path)
            else:
                if entry.path.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(entry.path)
                    label = os.path.basename(os.path.dirname(entry.path))  # Lấy tên thư mục gần nhất làm nhãn
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        label = self.label_map[label]  # Chuyển nhãn thành số nguyên

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)  # Trả về label dưới dạng tensor


# Khởi tạo các transform 
train_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Tạo đối tượng Dataset
train_dataset = CustomDataset(root_dir=r'D:\New_IVSR\data', transform=train_transforms)
val_dataset = CustomDataset(root_dir=r'D:\New_IVSR\data', transform=train_transforms) # Giả sử bạn có val_dataset

# Chia theo tỉ lệ 80/20
dataset_size = len(train_dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])



# Tạo DataLoader từ dataset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # Tạo val_loader
