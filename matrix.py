import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score
import seaborn as sns
import numpy as np  # Thêm import numpy

# Nạp lại lịch sử huấn luyện từ file 'history.pkl'
with open('history.pkl', 'rb') as f:
    history = pickle.load(f)

def plot_accuracy(history):
    """
    Vẽ biểu đồ độ chính xác (accuracy) từ lịch sử huấn luyện của mô hình.
    """
    y_true = history['y_true']
    y_pred = history['y_pred']
    
    # Tính toán accuracy cho từng epoch
    accuracy = []
    for i in range(len(y_true)):
        accuracy.append(accuracy_score(y_true[i], y_pred[i]))

    # Vẽ đồ thị accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(accuracy, label='Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(conf_matrix, classes):
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()

# Gọi hàm vẽ đồ thị
if __name__ == "__main__":
    plot_accuracy(history)

    # Hiển thị confusion matrix
    conf_matrix = history['conf_matrix']
    num_classes = conf_matrix.shape[0]
    plot_confusion_matrix(conf_matrix, classes=[f'Lớp {i}' for i in range(num_classes)])