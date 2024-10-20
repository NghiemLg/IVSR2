from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler
import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np



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

# Đường dẫn đến thư mục chứa dữ liệu
data_dir = r'D:\New_IVSR\data\train'

# Tạo đối tượng Dataset từ thư mục train
train_dataset = CustomDataset(root_dir=os.path.join(data_dir, 'train'), transform=train_transforms)

# Chia dữ liệu thành train/validation/test với tỷ lệ 80/10/10
dataset_size = len(train_dataset)
indices = list(range(dataset_size))
np.random.shuffle(indices)
train_split = int(np.floor(0.8 * dataset_size))
val_split = int(np.floor(0.1 * dataset_size))
train_indices, val_indices, test_indices = indices[:train_split], indices[train_split:train_split+val_split], indices[train_split+val_split:]

# Tạo SubsetRandomSampler cho tập validation và test
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

# Tạo DataLoader cho từng tập dữ liệu
train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
val_loader = DataLoader(train_dataset, batch_size=32, sampler=val_sampler)
test_loader = DataLoader(train_dataset, batch_size=32, sampler=test_sampler)