import os
import sys
from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

code_dir = '/root/autodl-tmp/zyk_drought_monitor'
if code_dir not in sys.path:
    sys.path.append(code_dir)
os.chdir(code_dir)

from models.baseline.convlstm import ConvLSTM # atte-convlstm
from models.baseline.traj_gru import TrajGRU # TrajGRU
from trainer import Trainer
from configs.config import model_params

ACTIVE_MODEL = 'convlstm'  # 可选: 'atte-convlstm' 或 'traj_gru'
LABEL_MODE = 'threshold'      # 可选: 'kmeans' 或 'threshold'
BATCH_SIZE = 16
TRAIN_YEARS = [2021, 2022, 2023]
VAL_YEAR = 2024
TEST_YEAR = 2025
X_CANDIDATE_DIRS = [
    '/root/autodl-tmp/data_proc',
    '/root/autodl-tmp/data_proc/data_proc',
    '/content/drive/MyDrive/GEE_Drought_Project/data_proc',
    '/content/drive/MyDrive/drought_monitor/data_proc',
]
Y_CANDIDATE_DIRS = [
    '/root/autodl-tmp/data_proc',
    '/root/autodl-tmp/data_proc/data_proc',
    '/content/drive/MyDrive/GEE_Drought_Project/data_proc',
    '/content/drive/MyDrive/drought_monitor/data_proc',
]
OUTPUT_DIR = '/root/autodl-tmp/zyk_drought_monitor/data'


class SimpleDataWrapper:
    def __init__(self, train_loader, val_loader, test_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def generate(self, mode):
        if mode == 'train':
            return self.train_loader
        if mode == 'val':
            return self.val_loader
        if mode == 'test':
            return self.test_loader
        raise ValueError(f'未知模式: {mode}')

    def num_iter(self, mode):
        if mode == 'train':
            return len(self.train_loader)
        if mode == 'val':
            return len(self.val_loader)
        if mode == 'test':
            return len(self.test_loader)
        raise ValueError(f'未知模式: {mode}')



def find_existing_file(candidate_dirs, candidate_names: list[str]) -> str:
    checked_paths = []
    for directory in candidate_dirs:
        for name in candidate_names:
            path = os.path.join(directory, name)
            checked_paths.append(path)
            if os.path.exists(path):
                return path

    checked_text = '\n'.join(f'  - {path}' for path in checked_paths)
    raise FileNotFoundError(f'未找到任何候选文件，请检查数据路径或文件名：\n{checked_text}')



def resolve_x_path(year: int) -> str:
    candidate_names = [
        f'dataset_X_{year}.pt',
        'dataset_X.pt' if year == TRAIN_YEARS[0] else f'dataset_X_{year}.pt',
    ]
    return find_existing_file(X_CANDIDATE_DIRS, candidate_names)



def resolve_y_path(year: int, label_mode: str) -> str:
    if label_mode == 'kmeans':
        candidate_names = [
            f'dataset_Y_{year}.pt',
            'dataset_Y.pt' if year == TRAIN_YEARS[0] else f'dataset_Y_{year}.pt',
        ]
    elif label_mode == 'threshold':
        candidate_names = [
            f'dataset_Y_{year}_threshold.pt',
            'dataset_Y_threshold.pt',
        ]
    else:
        raise ValueError(f'不支持的 LABEL_MODE: {label_mode}')

    return find_existing_file(Y_CANDIDATE_DIRS, candidate_names)



def load_year_pair(year: int, label_mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    x_path = resolve_x_path(year)
    y_path = resolve_y_path(year, label_mode)
    print(f'加载 {year} 年特征: {x_path}')
    print(f'加载 {year} 年标签({label_mode}): {y_path}')
    x_tensor = torch.load(x_path, map_location='cpu')
    y_tensor = torch.load(y_path, map_location='cpu')
    return x_tensor, y_tensor



def compute_class_weights(y_train: torch.Tensor, device: torch.device) -> torch.Tensor:
    all_labels = y_train.flatten().long()
    class_counts = torch.bincount(all_labels, minlength=4)
    class_counts = torch.clamp(class_counts, min=1)
    class_weights = len(all_labels) / (len(class_counts) * class_counts.float())
    return class_weights.to(device)



def build_model(active_model: str, device: torch.device):
    if active_model == 'convlstm':
        model_config = model_params['convlstm']
        model = ConvLSTM(
            input_size=model_config['core']['input_size'],
            window_in=model_config['core']['window_in'],
            num_layers=model_config['core']['num_layers'],
            encoder_params=model_config['core']['encoder_params'],
            input_attn_params=model_config['core']['input_attn_params'],
            device=device,
        )
        save_name = f'drought_convlstm_best_{LABEL_MODE}.pth'
    elif active_model == 'traj_gru':
        model_config = model_params['traj_gru']
        model = TrajGRU(
            input_size=model_config['core']['input_size'],
            window_in=model_config['core']['window_in'],
            window_out=model_config['core']['window_out'],
            encoder_params=model_config['core']['encoder_params'],
            decoder_params=model_config['core']['decoder_params'],
            num_classes=model_config['core']['num_classes'],
            device=device,
        ).to(device)
        save_name = f'drought_trajgru_best_{LABEL_MODE}.pth'
    else:
        raise ValueError(f'不支持的 ACTIVE_MODEL: {active_model}')

    return model, model_config, save_name



def main():
    print('正在加载历史数据...')
    print(f'使用模型: {ACTIVE_MODEL.upper()}')
    print(f'标签方案: {LABEL_MODE}')

    train_pairs = [load_year_pair(year, LABEL_MODE) for year in TRAIN_YEARS]
    x_train = torch.cat([pair[0] for pair in train_pairs], dim=0)
    y_train = torch.cat([pair[1] for pair in train_pairs], dim=0)
    x_val, y_val = load_year_pair(VAL_YEAR, LABEL_MODE)
    x_test, y_test = load_year_pair(TEST_YEAR, LABEL_MODE)

    print(f'训练集形状: X={x_train.shape}, Y={y_train.shape}')
    print(f'验证集形状: X={x_val.shape}, Y={y_val.shape}')
    print(f'测试集形状: X={x_test.shape}, Y={y_test.shape}')

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=BATCH_SIZE, shuffle=False)
    batch_generator = SimpleDataWrapper(train_loader, val_loader, test_loader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'当前使用计算设备: {device}')

    print('正在计算损失函数类别权重...')
    class_weights = compute_class_weights(y_train, device)
    print(f'各干旱等级权重: {class_weights.cpu().numpy()}')

    model, model_config, save_name = build_model(ACTIVE_MODEL, device)
    trainer_config = model_config['trainer']
    trainer = Trainer(
        num_epochs=trainer_config['num_epochs'],
        early_stop_tolerance=trainer_config['early_stop_tolerance'],
        clip=trainer_config['clip'],
        optimizer=trainer_config['optimizer'],
        learning_rate=0.001,
        weight_decay=trainer_config['weight_decay'],
        momentum=trainer_config['momentum'],
        device=device,
        class_weights=class_weights,
    )

    print('开始模型训练...')
    losses, best_train, best_val = trainer.train(model=model, batch_generator=batch_generator)

    print(f'\n开始在 {TEST_YEAR} 年未见过的测试集上进行最终评估...')
    test_loss, test_metrics = trainer.evaluate(model=model, batch_generator=batch_generator)
    print(f'测试集结果 -> Loss: {test_loss:.5f}, ' + trainer.get_metric_string(test_metrics))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, save_name)
    torch.save(model.state_dict(), save_path)
    print(f'训练完成，最优 {ACTIVE_MODEL.upper()} 模型权重已保存至：{save_path}！')
    print(f'本次训练标签方案：{LABEL_MODE}')
    print(f'最佳训练指标：{best_train}')
    print(f'最佳验证指标：{best_val}')
    print(f'测试指标：{test_metrics}')
    print(f'损失曲线记录长度：train={len(losses[0])}, val={len(losses[1])}')


if __name__ == '__main__':
    main()
