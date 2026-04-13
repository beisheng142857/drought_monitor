import torch
import os
import sys
from torch.utils.data import TensorDataset, DataLoader, random_split

code_dir = '/content/drive/MyDrive/zyk_drought_monitor'
# 将代码目录加入系统的环境变量中，这样 Python 就能找到 models 文件夹了
if code_dir not in sys.path:
    sys.path.append(code_dir)

os.chdir(code_dir)

# 导入你修改后的模型和训练器
# 确保你的 Colab 当前工作目录在 spatio-temporal-weather-forecasting 文件夹下
from models.baseline.convlstm import ConvLSTM
from trainer import Trainer
from configs.config import model_params


# ==========================================
# 1. 加载我们准备好的张量数据
# ==========================================
base_dir = '/content/drive/MyDrive/zyk_drought_monitor/data' 
print("正在加载并合并多份历史数据...")

# 2021年
X_1 = torch.load(os.path.join(base_dir, 'dataset_X_2021.pt')) 
Y_1 = torch.load(os.path.join(base_dir, 'dataset_Y_2021.pt'))

# 2022年
X_2 = torch.load(os.path.join(base_dir, 'dataset_X_2022.pt')) 
Y_2 = torch.load(os.path.join(base_dir, 'dataset_Y_2022.pt'))

# 2023年
X_3 = torch.load(os.path.join(base_dir, 'dataset_X_2023.pt')) 
Y_3 = torch.load(os.path.join(base_dir, 'dataset_Y_2023.pt'))

# 2024年
X_4 = torch.load(os.path.join(base_dir, 'dataset_X_2024.pt')) 
Y_4 = torch.load(os.path.join(base_dir, 'dataset_Y_2024.pt'))

# # 2025年
# X_5 = torch.load(os.path.join(base_dir, 'dataset_X_2025.pt')) 
# Y_5 = torch.load(os.path.join(base_dir, 'dataset_Y_2025.pt'))

# 3. 核心魔法：使用 torch.cat 将它们在第 0 维（Batch 也就是样本数量维度）无缝拼接！
X = torch.cat([X_1, X_2, X_3], dim=0)
Y = torch.cat([Y_1, Y_2, Y_3], dim=0)

print(f"数据合并成功")
print(f"最终喂给模型的 X 形状: {X.shape}")
print(f"最终喂给模型的 Y 形状: {Y.shape}")

# 打包为 Dataset 并划分训练集/验证集 (8:2)
dataset = TensorDataset(X, Y)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建极简的 DataLoader (128x128图块，batch_size设为16比较合适)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ==========================================
# 2. 编写一个简单的包装器，适配原仓库的 Trainer
# ==========================================
class SimpleDataWrapper:
    """把 PyTorch 的 DataLoader 伪装成 Trainer 需要的 batch_generator"""
    def __init__(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.normalizer = None # 我们不需要它自带的归一化器

    def generate(self, mode):
        if mode == 'train': return self.train_loader
        if mode == 'val': return self.val_loader

    def num_iter(self, mode):
        if mode == 'train': return len(self.train_loader)
        if mode == 'val': return len(self.val_loader)

# 实例化包装器
batch_generator = SimpleDataWrapper(train_loader, val_loader)

# ==========================================
# 3. 初始化模型与配置
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用计算设备: {device}")

# 从配置中提取 convlstm 的参数
convlstm_config = model_params['convlstm']

model = ConvLSTM(
    input_size=convlstm_config['core']['input_size'],  # (128, 128)
    window_in=convlstm_config['core']['window_in'],    # 5个月
    num_layers=convlstm_config['core']['num_layers'],  # 2层
    encoder_params=convlstm_config['core']['encoder_params'],
    device=device
)

# 初始化 Trainer
trainer_config = convlstm_config['trainer']
trainer = Trainer(
    num_epochs=trainer_config['num_epochs'],
    early_stop_tolerance=trainer_config['early_stop_tolerance'],
    clip=trainer_config['clip'],
    optimizer=trainer_config['optimizer'],
    learning_rate=0.001, # 取0.001
    weight_decay=trainer_config['weight_decay'],
    momentum=trainer_config['momentum'],
    device=device
)

# ==========================================
# 4. 开始训练干旱监测模型
# ==========================================
print("开始模型训练...")
losses, best_train, best_val = trainer.train(model=model, batch_generator=batch_generator)
train_loss = losses[0]
val_loss = losses[1]

# 保存训练好的模型权重
torch.save(model.state_dict(), os.path.join(base_dir, 'drought_convlstm_best_2021_2024.pth'))
print("训练完成，模型权重已保存！")
