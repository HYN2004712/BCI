import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import mne
from sklearn.metrics import accuracy_score

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

class ChannelAttention(nn.Module):
    """通道注意力模块（Tensor Product实现）"""

    def __init__(self, C, D=9):
        super(ChannelAttention, self).__init__()
        # 可学习参数矩阵 (D × C)
        self.weight = nn.Parameter(torch.Tensor(D, C))
        nn.init.normal_(self.weight, mean=0.0, std=0.01)  # 论文中的初始化方法

    def forward(self, x):
        """
        输入: x (batch, 1, C, T)
        输出: x' (batch, D, 1, T)
        """
        batch_size, _, C, T = x.shape

        # 张量积操作 (公式2)
        x = x.squeeze(1)  # (batch, C, T)
        x = x.permute(0, 2, 1)  # (batch, T, C)
        x_prime = torch.matmul(x, self.weight.T)  # (batch, T, D)
        x_prime = x_prime.permute(0, 2, 1)  # (batch, D, T)
        x_prime = x_prime.unsqueeze(2)  # (batch, D, 1, T)

        return x_prime


class DepthAttention(nn.Module):
    def __init__(self, D, k=7):  # 关键修改：k=7（奇数）
        super(DepthAttention, self).__init__()
        self.conv = nn.Conv1d(D, D, kernel_size=k, padding=k // 2, groups=1)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, F_input):
        # 半全局池化
        pooled = F_input.mean(dim=2, keepdim=False)  # (batch, D, T')

        # 卷积（保持尺寸）
        conv_out = self.conv(pooled)

        # Softmax加权
        softmax_weights = F.softmax(conv_out, dim=1)
        softmax_weights = softmax_weights.unsqueeze(2)

        # Hadamard乘积
        output = F_input * softmax_weights
        return output

# 2. 构建 LMDA_Net 模型（PyTorch 实现）
class LMDA_Net(nn.Module):
    """修复空间卷积维度问题后的 LMDA-Net"""

    def __init__(self, C, T, n_classes, D=9, k=7):
        super(LMDA_Net, self).__init__()
        self.C = C  # 新增：保存原始通道数
        self.D = D
        self.k = k

        # 1. 通道注意力模块
        self.channel_attn = ChannelAttention(C, D=self.D)

        # 2. 时间卷积层
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(D, 24, kernel_size=(1, 75), padding=(0, 37), bias=False),
            nn.BatchNorm2d(24),
            nn.Conv2d(24, 24, kernel_size=(1, 175), padding=(0, 87), groups=24, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU()
        )

        # 3. 深度注意力模块
        self.depth_attn = DepthAttention(D=24, k=self.k)

        # 4. 空间卷积层（修正输入维度）
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(24, 9, kernel_size=(self.C, 1), padding=0, bias=False),  # 高度=C
            nn.BatchNorm2d(9),
            nn.Conv2d(9, 9, kernel_size=(1, 11), padding=5, groups=9, bias=False),
            nn.BatchNorm2d(9),
            nn.GELU(),
            nn.AvgPool2d((1, 4), stride=(1, 4)),
            nn.Dropout(0.65)
        )

        # 5. 分类器
        self.fc = nn.Linear(self._calculate_fc_input(C, T), n_classes)

    def _calculate_fc_input(self, C, T):
        dummy = torch.randn(1, 1, C, T)
        x = self.channel_attn(dummy)  # (1, D=9, 1, T)
        x = self.temporal_conv(x)  # (1, 24, 1, T')
        x = self.depth_attn(x)  # (1, 24, 1, T')

        # 扩展维度以匹配空间卷积输入
        x = x.repeat(1, 1, C, 1)  # (1, 24, C, T')
        x = self.spatial_conv(x)  # (1, 9, 1, T'')
        return x.view(1, -1).size(1)

    def forward(self, x):
        # 输入维度: (batch, 1, C, T)
        x = self.channel_attn(x)  # (batch, D=9, 1, T)
        x = self.temporal_conv(x)  # (batch, 24, 1, T')
        x = self.depth_attn(x)  # (batch, 24, 1, T')

        # 扩展通道维度以匹配空间卷积
        x = x.repeat(1, 1, self.C, 1)  # (batch, 24, C, T')
        x = self.spatial_conv(x)  # (batch, 9, 1, T'')
        x = x.flatten(1)
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
def train_eegnet(X_train, y_train, X_test, y_test, n_channels, n_times, n_classes, epochs=300, batch_size=32):
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
    model = LMDA_Net(n_channels, n_times, n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
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
        train_number = random.randint(0, 4)
        test_number = random.randint(0, 2)
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
        model, accuracy = train_eegnet(X_train, y_train, X_test, y_test, n_channels, n_times, n_classes)

        # 输出结果
        print(f"模型 {i+1} 的测试准确率: {accuracy * 100:.2f}%")
        print("=" * 50)