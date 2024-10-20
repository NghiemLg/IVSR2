import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import train_loader, val_loader
import torch.nn.functional as F
from tqdm import tqdm
import dill
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import model as myCNN

def plot_confusion_matrix(conf_matrix, classes):
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()

def train_model(model, criterion, optimizer, epochs, train_loader, val_loader, num_classes):
    best_val_acc = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    all_y_true = []
    all_y_pred = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{epochs}")
            for images, labels in tepoch:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                tepoch.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        y_true = []  # Khai báo y_true và y_pred ở đây
        y_pred = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item() 

                y_true.append(labels.cpu().numpy())  # Lưu mảng nhãn
                y_pred.append(predicted.cpu().numpy())  # Lưu mảng dự đoán

        # Thêm y_true và y_pred của epoch hiện tại vào all_y_true và all_y_pred
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_accuracy = 100 * val_correct / val_total
        print(f'Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.2f}%')

        if val_epoch_accuracy > best_val_acc:
            best_val_acc = val_epoch_accuracy
            torch.save(model.state_dict(), 'best_model.pth', pickle_module=dill)
            print("Đã lưu mô hình tốt nhất vào best_model.pth")

        # Tính toán Confusion Matrix
        conf_matrix = confusion_matrix(np.concatenate(y_true), np.concatenate(y_pred))  # Nối các mảng trong y_true và y_pred
        print(f"Confusion Matrix:\n{conf_matrix}")
        plot_confusion_matrix(conf_matrix, classes=[f'Lớp {i}' for i in range(num_classes)])

    # Lưu lịch sử huấn luyện và confusion matrix (chỉ lưu một lần)
    history = {
        'y_true': all_y_true,
        'y_pred': all_y_pred,
        'conf_matrix': conf_matrix,  # Thêm confusion matrix vào history
    }

    with open('history.pkl', 'wb') as f:
        pickle.dump(history, f)

    torch.save(model.state_dict(), 'final_model.pth')
    print("Đã lưu mô hình vào final_model.pth")

    return all_y_true, all_y_pred  # Trả về all_y_true và all_y_pred

if __name__ == '__main__':
    num_classes = 8
    model = myCNN(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 1  # Tăng số lượng epochs

    y_true, y_pred = train_model(model, criterion, optimizer, epochs, train_loader, val_loader, num_classes)

    # Tính toán Accuracy Metric
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy Metric: {accuracy:.4f}')

