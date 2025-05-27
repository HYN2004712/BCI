import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import mne
from sklearn.metrics import accuracy_score

# 1.
#加载训练集文件
def process_train_file(file_path, tmin=1, tmax=4, event_id=None, bad_channels=None):
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

#加载测试集文件
def process_Val_GDF_file(file_path, tmin=1, tmax=4, event_id=None, bad_channels=None):
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
        event_id = {'783': 7}  # 测试集
    if bad_channels is None:
        bad_channels = ['EOG-left', 'EOG-central', 'EOG-right']

    raw = mne.io.read_raw_gdf(file_path, preload=True)
    events, _ = mne.events_from_annotations(raw)
    raw.info['bads'] += bad_channels
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
    # labels = epochs.events[:, -1] - 6  # 标签映射为 (1, 2, 3, 4)
    data = epochs.get_data()  # 形状 (n_epochs, n_channels, n_times)
    return data

def process_mat_file(file_path):
    # 读取 .mat 文件
    mat_data = scipy.io.loadmat(file_path)

    # 查看文件中的变量名（MATLAB工作区变量）
    print(mat_data.keys())  # 输出所有变量名，如 '__header__', '__version__', 'data', 'labels' 等

    # 提取具体变量
    labels = mat_data['classlabel']
    return labels

# 2. 构建 EEGNet 模型（PyTorch 实现）
class ChannelAttention(nn.Module):
    """通道注意力机制"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 平均池化分支
        avg_out = self.fc(self.avg_pool(x).squeeze())
        # 最大池化分支
        max_out = self.fc(self.max_pool(x).squeeze())
        # 合并注意力权重
        scale = torch.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class EnhancedEEGNet(nn.Module):
    def __init__(self, n_channels, n_times, n_classes):
        super().__init__()

        # Block 1: 时空特征提取
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, (1, 64), padding=(0, 32), bias=False),  # 时间卷积
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (n_channels, 1), groups=32, bias=False),  # 空间卷积（压缩空间维度到1）
            nn.BatchNorm2d(64),
            ChannelAttention(64),  # 通道注意力
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  # 时间下采样4倍
            nn.Dropout(0.5)
        )

        # Block 2: 深层特征提取
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, (1, 16), groups=64, padding=(0, 8), bias=False),  # 深度卷积
            nn.Conv2d(64, 128, 1, bias=False),  # 逐点卷积
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),  # 时间下采样8倍（总下采样32倍）
            nn.Dropout(0.5)
        )

        # 残差路径（匹配主路径维度）
        self.residual_path = nn.Sequential(
            nn.Conv2d(1, 128, (n_channels, 1)),  # 空间卷积压缩维度
            nn.BatchNorm2d(128),
            nn.AvgPool2d((1, 4)),  # 匹配block1下采样
            nn.AvgPool2d((1, 8))  # 匹配block2下采样
        )

        # 动态全连接层
        self.fc = nn.Linear(self._get_fc_input(n_channels, n_times), n_classes)

    def _get_fc_input(self, n_channels, n_times):
        """动态计算全连接层输入维度"""
        dummy = torch.randn(1, 1, n_channels, n_times)
        main_path = self.block2(self.block1(dummy))
        residual = self.residual_path(dummy)
        combined = main_path + residual
        return combined.view(1, -1).size(1)

    def forward(self, x):
        # 残差连接
        residual = self.residual_path(x)
        x = self.block1(x)
        x = self.block2(x)
        x = x + residual

        # 分类层
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

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
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(test_labels, dtype=torch.long)
    )

# 4. 训练函数（独立训练每个文件对）
def train_eegnet(X_train, y_train, X_test, y_test, n_channels, n_times, n_classes, epochs=200, batch_size=32):
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
    model = EnhancedEEGNet(n_channels, n_times, n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    train_dataset = TensorDataset(X_train, y_train - 1)  # 标签调整 [1,4] -> [0,3]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()

        # 打印训练信息
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_train / total_train
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc*100:.2f}%")

        # 测试集评估
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted_test = torch.max(outputs, 1)
            predicted_test = predicted_test + 1  # 标签恢复 [0,3] -> [1,4]
            test_acc = accuracy_score(y_test.numpy(), predicted_test.numpy())
            print(f"Epoch [{epoch+1}/{epochs}], Test Acc: {test_acc*100:.2f}%")
            print("-" * 50)

    # 最终测试准确率
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        predicted = predicted + 1
        accuracy = accuracy_score(y_test.numpy(), predicted.numpy())
    return model, accuracy
#within-subject
if __name__ == "__main__":
    file_pairs = [
        (
            '/Users/heyunuo/Desktop/pythonProject1/BCICIV_2a_gdf/A01T_train.gdf',
            '/Users/heyunuo/Desktop/pythonProject1/BCICIV_2a_gdf/A01E_val.gdf',
            '/Users/heyunuo/Desktop/pythonProject1/Desktop/true_labels/A01E.mat',
        ),
        (
            '/Users/heyunuo/Desktop/pythonProject1/BCICIV_2a_gdf/A02T_train.gdf',
            '/Users/heyunuo/Desktop/pythonProject1/BCICIV_2a_gdf/A02E_val.gdf',
            '/Users/heyunuo/Desktop/pythonProject1/Desktop/true_labels/A02E.mat',
        ),
        (
            '/Users/heyunuo/Desktop/pythonProject1/BCICIV_2a_gdf/A03T_train.gdf',
            '/Users/heyunuo/Desktop/pythonProject1/BCICIV_2a_gdf/A03E_val.gdf',
            '/Users/heyunuo/Desktop/pythonProject1/Desktop/true_labels/A03E.mat',
        ),
        # (
        #     '/Users/heyunuo/Desktop/pythonProject1/BCICIV_2a_gdf/A04T_train.gdf',
        #     '/Users/heyunuo/Desktop/pythonProject1/BCICIV_2a_gdf/A04E_val.gdf',
        #     '/Users/heyunuo/Desktop/pythonProject1/Desktop/true_labels/A04E.mat',
        # ),
        (
            '/Users/heyunuo/Desktop/pythonProject1/BCICIV_2a_gdf/A05T_train.gdf',
            '/Users/heyunuo/Desktop/pythonProject1/BCICIV_2a_gdf/A05E_val.gdf',
            '/Users/heyunuo/Desktop/pythonProject1/Desktop/true_labels/A05E.mat',
        ),
        (
            '/Users/heyunuo/Desktop/pythonProject1/BCICIV_2a_gdf/A06T_train.gdf',
            '/Users/heyunuo/Desktop/pythonProject1/BCICIV_2a_gdf/A06E_val.gdf',
            '/Users/heyunuo/Desktop/pythonProject1/Desktop/true_labels/A06E.mat',
        ),
        (
            '/Users/heyunuo/Desktop/pythonProject1/BCICIV_2a_gdf/A07T_train.gdf',
            '/Users/heyunuo/Desktop/pythonProject1/BCICIV_2a_gdf/A07E_val.gdf',
            '/Users/heyunuo/Desktop/pythonProject1/Desktop/true_labels/A07E.mat',
        ),
        (
            '/Users/heyunuo/Desktop/pythonProject1/BCICIV_2a_gdf/A08T_train.gdf',
            '/Users/heyunuo/Desktop/pythonProject1/BCICIV_2a_gdf/A08E_val.gdf',
            '/Users/heyunuo/Desktop/pythonProject1/Desktop/true_labels/A08E.mat',
        ),
        (
            '/Users/heyunuo/Desktop/pythonProject1/BCICIV_2a_gdf/A09T_train.gdf',
            '/Users/heyunuo/Desktop/pythonProject1/BCICIV_2a_gdf/A09E_val.gdf',
            '/Users/heyunuo/Desktop/pythonProject1/Desktop/true_labels/A09E.mat',
        )
    ]
    for i, (train_path,val_path, mat_path) in enumerate(file_pairs):
        print(f"处理文件对 {i + 1}: ")

        # 加载数据
        train_data, train_labels = process_train_file(train_path)
        test_labels = process_mat_file(mat_path)
        test_data = process_Val_GDF_file(val_path)

        # 预处理
        X_train, y_train, X_test, y_test = preprocess_pair(train_data, train_labels, test_data, test_labels)

        # 获取模型参数
        n_channels = X_train.shape[2]
        n_times = X_train.shape[3]
        n_classes = 4

        # 训练模型
        print(f"训练模型 {i + 1}...")
        model, accuracy = train_eegnet(X_train, y_train, X_test, y_test, n_channels, n_times, n_classes)

        # 输出结果
        print(f"模型 {i + 1} 的测试准确率: {accuracy * 100:.2f}%")
        print("=" * 50)

