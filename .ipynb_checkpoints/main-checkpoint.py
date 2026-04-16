import torch
import os
import sys
from torch.utils.data import TensorDataset, DataLoader

code_dir = '/root/autodl-tmp/zyk_drought_monitor'
if code_dir not in sys.path:
    sys.path.append(code_dir)
os.chdir(code_dir)

from models.baseline.convlstm import ConvLSTM
from trainer import Trainer
from configs.config import model_params

# ==========================================
# 1. 加载并按【时间顺序】划分数据
# ==========================================
base_dir = '/root/autodl-tmp/zyk_drought_monitor/data' 
print("正在加载历史数据...")

X_1, Y_1 = torch.load(os.path.join(base_dir, 'dataset_X_2021.pt')), torch.load(os.path.join(base_dir, 'dataset_Y_2021.pt'))
X_2, Y_2 = torch.load(os.path.join(base_dir, 'dataset_X_2022.pt')), torch.load(os.path.join(base_dir, 'dataset_Y_2022.pt'))
X_3, Y_3 = torch.load(os.path.join(base_dir, 'dataset_X_2023.pt')), torch.load(os.path.join(base_dir, 'dataset_Y_2023.pt'))
X_4, Y_4 = torch.load(os.path.join(base_dir, 'dataset_X_2024.pt')), torch.load(os.path.join(base_dir, 'dataset_Y_2024.pt'))

# 【优化】按年份划分，防止时间序列特征泄露
# 训练集: 2021-2022
X_train = torch.cat([X_1, X_2], dim=0)
Y_train = torch.cat([Y_1, Y_2], dim=0)
# 验证集: 2023 (用于早停)
X_val, Y_val = X_3, Y_3
# 测试集: 2024 (用于最终评估)
X_test, Y_test = X_4, Y_4

print(f"训练集形状: X={X_train.shape}, Y={Y_train.shape}")
print(f"验证集形状: X={X_val.shape}, Y={Y_val.shape}")
print(f"测试集形状: X={X_test.shape}, Y={Y_test.shape}")

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=16, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=16, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=16, shuffle=False)

# ==========================================
# 2. 【核心优化】计算类别权重 (缓解干旱类别不平衡)
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用计算设备: {device}")

print("正在计算损失函数类别权重...")
all_labels = Y_train.flatten().long()
class_counts = torch.bincount(all_labels)
# 权重计算公式: 总样本数 / (类别数 * 该类样本数)
# 这样样本越少的类别，权重越大
class_weights = len(all_labels) / (len(class_counts) * class_counts.float())
class_weights = class_weights.to(device)
print(f"各干旱等级权重: {class_weights.cpu().numpy()}")

# ==========================================
# 3. 初始化包装器、模型与配置
# ==========================================
class SimpleDataWrapper:
    def __init__(self, train_loader, val_loader, test_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def generate(self, mode):
        if mode == 'train': return self.train_loader
        if mode == 'val': return self.val_loader
        if mode == 'test': return self.test_loader

    def num_iter(self, mode):
        if mode == 'train': return len(self.train_loader)
        if mode == 'val': return len(self.val_loader)
        if mode == 'test': return len(self.test_loader)

batch_generator = SimpleDataWrapper(train_loader, val_loader, test_loader)

convlstm_config = model_params['convlstm']
model = ConvLSTM(
    input_size=convlstm_config['core']['input_size'],  
    window_in=convlstm_config['core']['window_in'],    
    num_layers=convlstm_config['core']['num_layers'],  
    encoder_params=convlstm_config['core']['encoder_params'],
    input_attn_params=convlstm_config['core']['input_attn_params'],
    device=device
)

trainer_config = convlstm_config['trainer']
trainer = Trainer(
    num_epochs=trainer_config['num_epochs'],
    early_stop_tolerance=trainer_config['early_stop_tolerance'],
    clip=trainer_config['clip'],
    optimizer=trainer_config['optimizer'],
    learning_rate=0.001, 
    weight_decay=trainer_config['weight_decay'],
    momentum=trainer_config['momentum'],
    device=device,
    class_weights=class_weights # 传入类别权重
)

# ==========================================
# 4. 开始训练与最终测试
# ==========================================
print("开始模型训练...")
losses, best_train, best_val = trainer.train(model=model, batch_generator=batch_generator)

print("\n开始在 2024 年未见过的测试集上进行最终评估...")
test_loss, test_metrics = trainer.evaluate(model=model, batch_generator=batch_generator)
print(f"测试集结果 -> Loss: {test_loss:.5f}, " + trainer.get_metric_string(test_metrics))

torch.save(model.state_dict(), os.path.join(base_dir, 'drought_convlstm_best_with_attn.pth'))
print("训练完成，最优模型权重已保存！")