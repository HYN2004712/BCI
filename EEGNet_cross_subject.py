import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import mne
from sklearn.metrics import accuracy_score


# --------------------- 中心损失定义 ---------------------
class CenterLoss(nn.Module):
    """约束同类样本特征靠近类别中心的损失"""

    def __init__(self, num_classes, feat_dim, alpha=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.alpha = alpha  # 中心更新速率

        # 初始化可学习的类别中心
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        """
        参数:
            features: 特征向量 [batch_size, feat_dim]
            labels: 真实标签 [batch_size]
        """
        batch_size = features.size(0)

        # 获取当前批次样本对应的中心 [batch_size, feat_dim]
        centers_batch = self.centers[labels]

        # 计算特征与中心的L2距离
        loss = 0.5 * torch.sum((features - centers_batch) ** 2) / batch_size

        # 动态更新中心（可选）
        with torch.no_grad():
            for cls in torch.unique(labels):
                cls_features = features[labels == cls]
                if cls_features.size(0) > 0:
                    self.centers.data[cls] = (1 - self.alpha) * self.centers.data[cls] + \
                                             self.alpha * cls_features.mean(dim=0)
        return loss

# 1. 加载和预处理单个EEG文件
def process_single_file(file_path, tmin=1, tmax=4, event_id=None, bad_channels=None):
    """
    处理单个EEG文件的函数。

    参数:
        file_path (str): GDF文件路径。
        tmin (float): Epoch的起始时间（相对于事件的时间，单位为秒）。
        tmax (float): Epoch的结束时间（相对于事件的时间，单位为秒）。
        event_id (dict): 事件ID映射字典，例如 {'769': 7, '770': 8, '771': 9, '772': 10}。
        bad_channels (list): 需要剔除的坏通道列表。

    返回:
        data (np.array): EEG数据，形状为 (n_epochs, n_channels, n_times)。
        labels (np.array): 标签数据，形状为 (n_epochs,)。
    """
    if event_id is None:
        event_id = {'769': 7, '770': 8, '771': 9, '772': 10}  # 左、右、足、舌
    if bad_channels is None:
        bad_channels = ['EOG-left', 'EOG-central', 'EOG-right']

    raw = mne.io.read_raw_gdf(file_path, preload=True)
    events, _ = mne.events_from_annotations(raw)
    raw.info['bads'] += bad_channels
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
    labels = epochs.events[:, -1] - 6  # 标签映射为 (1, 2, 3, 4)
    data = epochs.get_data()  # 形状 (n_epochs, n_channels, n_times)
    return data, labels

# 2. 构建 EEGNet 模型（PyTorch 实现）
class EEGNetWithCenterLoss(nn.Module):
    def __init__(self, n_channels, n_times, n_classes, dropout_rate=0.5):
        super().__init__()
        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, (1, 64), padding=(0, 32), bias=False),  # 32→64
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, (n_channels, 1), groups=64, bias=False),  # 32→64→128
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )
        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 128, (1, 32), padding=(0, 16), groups=128, bias=False),  # 64→128
            nn.Conv2d(128, 256, 1, bias=False),  # 128→256
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate)
        )
        # 特征维度计算
        self.feat_dim = self._get_feat_dim(n_channels, n_times)
        # 分类层
        self.fc = nn.Linear(self.feat_dim, n_classes)

    def _get_feat_dim(self, n_channels, n_times):
        dummy = torch.randn(1, 1, n_channels, n_times)
        return self.block2(self.block1(dummy)).view(1, -1).size(1)

    def forward(self, x):
        # 特征提取
        x = self.block1(x)
        x = self.block2(x)
        features = x.view(x.size(0), -1)  # [batch, feat_dim]

        # 分类结果
        outputs = self.fc(features)
        return outputs, features  # 同时返回输出和特征

# 3. 数据预处理（针对单个训练文件和测试文件）
def preprocess_pair(train_data, train_labels, test_data, test_labels):
    """
    处理训练和测试文件对，转换为PyTorch张量。

    参数:
        train_data (np.array): 训练数据，形状 (n_epochs, n_channels, n_times).
        train_labels (np.array): 训练标签，形状 (n_epochs,).
        test_data (np.array): 测试数据，形状 (n_epochs, n_channels, n_times).

    返回:
        X_train (torch.Tensor): 训练数据，形状 (n_epochs, 1, n_channels, n_times).
        y_train (torch.Tensor): 训练标签，形状 (n_epochs,).
        X_test (torch.Tensor): 测试数据，形状 (n_epochs, 1, n_channels, n_times).
        y_test (torch.Tensor): 测试标签，形状 (n_epochs,).
    """
    # 调整维度并转换为张量
    X_train = np.expand_dims(train_data, axis=1)
    X_test = np.expand_dims(test_data, axis=1)

    # 计算训练数据的均值和标准差
    mean = X_train.mean(axis=(0, 3), keepdims=True)
    std = X_train.std(axis=(0, 3), keepdims=True)

    # 应用标准化
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(test_labels, dtype=torch.long)
    )

# 4. 训练函数（独立训练每个文件对）
def train_eegnet(X_train, y_train, X_test, y_test, n_channels, n_times, n_classes,
                epochs=300, batch_size=32, lambda_center=0.1):
    """
    训练EEGNet模型（独立处理每个文件对）。

    参数:
        X_train (torch.Tensor): 训练数据。
        y_train (torch.Tensor): 训练标签。
        X_test (torch.Tensor): 测试数据。
        y_test (torch.Tensor): 测试标签。
        n_channels (int): EEG通道数。
        n_times (int): 时间点数。
        n_classes (int): 分类类别数。
        epochs (int): 训练轮数。
        batch_size (int): 批量大小。

    返回:
        model (torch.nn.Module): 训练好的模型。
        accuracy (float): 测试集准确率。
    """
    model = EEGNetWithCenterLoss(n_channels, n_times, n_classes)
    criterion_cls = nn.CrossEntropyLoss()  # 分类损失
    criterion_center = CenterLoss(n_classes, model.feat_dim)  # 中心损失

    # 优化器需要同时更新模型参数和中心参数
    optimizer = optim.Adam([
        {'params': model.parameters()},
        {'params': criterion_center.parameters()}
    ], lr=0.001, weight_decay=1e-4)

    # 数据加载
    train_dataset = TensorDataset(X_train, y_train - 1)  # 标签调整为0-3
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_acc = 0.0
    for epoch in range(epochs):
        # ================== 训练阶段 ==================
        model.train()
        running_cls_loss = 0.0
        running_center_loss = 0.0
        correct_train = 0
        total_train = 0

        for x, y in train_loader:
            optimizer.zero_grad()

            # 前向传播
            outputs, features = model(x)

            # 计算损失
            cls_loss = criterion_cls(outputs, y)
            center_loss = criterion_center(features, y)
            total_loss = cls_loss + lambda_center * center_loss

            # 反向传播
            total_loss.backward()
            optimizer.step()

            # 统计训练指标
            running_cls_loss += cls_loss.item()
            running_center_loss += center_loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += y.size(0)
            correct_train += (predicted == y).sum().item()

        # 计算训练指标
        train_cls_loss = running_cls_loss / len(train_loader)
        train_center_loss = running_center_loss / len(train_loader)
        train_total_loss = train_cls_loss + lambda_center * train_center_loss
        train_acc = 100. * correct_train / total_train

        # ================== 测试阶段 ==================
        model.eval()
        test_cls_loss = 0.0
        test_center_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            # 获取测试集全部数据
            outputs, features = model(X_test)

            # 计算分类损失
            test_cls_loss = criterion_cls(outputs, y_test - 1).item()

            # 计算中心损失（需要确保标签对齐）
            test_center_loss = criterion_center(features, y_test - 1).item()

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            total_test = y_test.size(0)
            correct_test = (predicted == (y_test - 1)).sum().item()
            test_acc = 100. * correct_test / total_test

        # 计算测试总损失
        test_total_loss = test_cls_loss + lambda_center * test_center_loss

        # ================== 输出信息 ==================
        print(f"\nEpoch [{epoch + 1}/{epochs}]")
        print(f"{'-' * 30}")
        print(f"| {'Metric':<20} | {'Training':<8} | {'Testing':<8} |")
        print(f"|{'-' * 22}|{'-' * 10}|{'-' * 10}|")
        print(f"| Class Loss        | {train_cls_loss:.4f}   | {test_cls_loss:.4f}   |")
        print(f"| Center Loss       | {train_center_loss:.4f}   | {test_center_loss:.4f}   |")
        print(f"| Total Loss        | {train_total_loss:.4f}   | {test_total_loss:.4f}   |")
        print(f"| Accuracy (%)      | {train_acc:6.2f}   | {test_acc:6.2f}   |")
        print(f"{'-' * 34}")

        # 更新最佳准确率
        if test_acc > best_acc:
            best_acc = test_acc

    return model, best_acc

# 主函数
if __name__ == "__main__":
    # 定义文件对路径（训练文件与测试文件一一对应）
    #1,2,3,5,6 训练集
    #7,8,9 测试集
    train_file = [
        '/Users/heyunuo/Desktop/pythonProject1/BCICIV_2a_gdf/A01T_train.gdf',
        '/Users/heyunuo/Desktop/pythonProject1/BCICIV_2a_gdf/A02T_train.gdf',
        '/Users/heyunuo/Desktop/pythonProject1/BCICIV_2a_gdf/A03T_train.gdf',
        '/Users/heyunuo/Desktop/pythonProject1/BCICIV_2a_gdf/A05T_train.gdf',
        '/Users/heyunuo/Desktop/pythonProject1/BCICIV_2a_gdf/A06T_train.gdf'
    ]
    test_file = [
        '/Users/heyunuo/Desktop/pythonProject1/BCICIV_2a_gdf/A07T_train.gdf',
        '/Users/heyunuo/Desktop/pythonProject1/BCICIV_2a_gdf/A08T_train.gdf',
        '/Users/heyunuo/Desktop/pythonProject1/BCICIV_2a_gdf/A09T_train.gdf'
    ]

    # 遍历每个文件对
    for i in range(10):
        print(f"第 {i+1}次 ")
        train_number = random.randint(0, len(train_file) - 1)  # 生成0-4的随机数
        test_number = random.randint(0, len(test_file) - 1)  # 生成0-2的随机数
        print(f"训练集 第{train_number+1}个")
        print(f"测试集 第{test_number+1}个")
        # 加载数据
        train_data, train_labels = process_single_file(train_file[train_number])
        test_data, test_labels = process_single_file(test_file[test_number])

        # 预处理
        X_train, y_train, X_test, y_test = preprocess_pair(train_data, train_labels, test_data, test_labels)

        # 获取模型参数
        n_channels = X_train.shape[2]
        n_times = X_train.shape[3]
        n_classes = 4

        # 训练模型
        print(f"训练模型 {i+1}...")
        model, acc = train_eegnet(X_train, y_train, X_test, y_test,n_channels, n_times, 4,lambda_center=0.1)

        # 输出结果
        print(f"模型 {i+1} 的测试准确率: {acc:.2f}%")
        print("=" * 50)