import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# 2. 构建 LMDANet 模型（PyTorch 实现）
class LMDA(nn.Module):
    """
    LMDA-Net for the paper
    """
    def __init__(self, chans=22, samples=1125, num_classes=4, depth=9, kernel=75, channel_depth1=24, channel_depth2=9,
                ave_depth=1, avepool=5):
        super(LMDA, self).__init__()
        self.ave_depth = ave_depth
        self.channel_weight = nn.Parameter(torch.randn(depth, 1, chans), requires_grad=True)
        nn.init.xavier_uniform_(self.channel_weight.data)


        self.time_conv = nn.Sequential(
            nn.Conv2d(depth, channel_depth1, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.Conv2d(channel_depth1, channel_depth1, kernel_size=(1, kernel),
                      groups=channel_depth1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.GELU(),
        )
        # self.avgPool1 = nn.AvgPool2d((1, 24))
        self.chanel_conv = nn.Sequential(
            nn.Conv2d(channel_depth1, channel_depth2, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.Conv2d(channel_depth2, channel_depth2, kernel_size=(chans, 1), groups=channel_depth2, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.GELU(),
        )

        self.norm = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, avepool)),
            # nn.AdaptiveAvgPool3d((9, 1, 35)),
            nn.Dropout(p=0.65),
        )

        # 定义自动填充模块
        out = torch.ones((1, 1, chans, samples))
        out = torch.einsum('bdcw, hdc->bhcw', out, self.channel_weight)
        out = self.time_conv(out)
        # out = self.avgPool1(out)
        out = self.chanel_conv(out)
        out = self.norm(out)
        n_out_time = out.cpu().data.numpy().shape
        print('In ShallowNet, n_out_time shape: ', n_out_time)
        self.classifier = nn.Linear(n_out_time[-1]*n_out_time[-2]*n_out_time[-3], num_classes)

    def EEGDepthAttention(self, x):
        # x: input features with shape [N, C, H, W]

        N, C, H, W = x.size()
        # K = W if W % 2 else W + 1
        k = 7
        adaptive_pool = nn.AdaptiveAvgPool2d((1, W))
        conv = nn.Conv2d(1, 1, kernel_size=(k, 1), padding=(k//2, 0), bias=True).to(x.device)  # original kernel k
        softmax = nn.Softmax(dim=-2)
        x_pool = adaptive_pool(x)
        x_transpose = x_pool.transpose(-2, -3)
        y = conv(x_transpose)
        y = softmax(y)
        y = y.transpose(-2, -3)
        return y * C * x

    def forward(self, x):
        x = torch.einsum('bdcw, hdc->bhcw', x, self.channel_weight)

        x_time = self.time_conv(x)  # batch, depth1, channel, samples_
        x_time = self.EEGDepthAttention(x_time)  # DA1

        x = self.chanel_conv(x_time)  # batch, depth2, 1, samples_
        x = self.norm(x)

        features = torch.flatten(x, 1)
        cls = self.classifier(features)
        return cls

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
    model = LMDA(n_channels, n_times, n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
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

