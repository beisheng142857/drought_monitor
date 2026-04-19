import copy
import os
import sys
from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

code_dir = '/root/autodl-tmp/zyk_drought_monitor'
if code_dir not in sys.path:
    sys.path.append(code_dir)
os.chdir(code_dir)

from configs.config import model_params
from models.baseline.convgru import ConvGRU
from models.baseline.convlstm import ConvLSTM
from models.baseline.traj_gru import TrajGRU
from trainer import Trainer

ACTIVE_MODEL = 'convgru'  # 可选: 'convlstm_attn'、'convlstm_no_attn'、'convgru'、'traj_gru'
LABEL_MODE = 'threshold'  # 可选: 'kmeans' 或 'threshold'
BATCH_SIZE = 16
TRAIN_YEARS = [2021, 2022]
VAL_YEAR = 2023
TEST_YEAR = 2024
FORECAST_INPUT_STEPS = 4  # 用前 4 个月预测第 5 个月旱情
FORECAST_TARGET_MONTH_INDEX = 4  # 预测窗口内第 5 个月，对应索引 4
X_CANDIDATE_DIRS = [
    '/root/autodl-tmp/data_proc',
    '/root/autodl-tmp/zyk_drought_monitor/data_V2',
    '/root/autodl-tmp/data_proc/data_proc',
    '/content/drive/MyDrive/GEE_Drought_Project/data_proc',
    '/content/drive/MyDrive/drought_monitor/data_proc',
]
Y_CANDIDATE_DIRS = [
    '/root/autodl-tmp/data_proc',
    '/root/autodl-tmp/zyk_drought_monitor/data_V2',
    '/root/autodl-tmp/data_proc/data_proc',
    '/content/drive/MyDrive/GEE_Drought_Project/data_proc',
    '/content/drive/MyDrive/drought_monitor/data_proc',
]
# 输出路径
OUTPUT_DIR = '/root/autodl-tmp/zyk_drought_monitor/data_V2'


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


def find_existing_file(candidate_dirs, candidate_names):
    checked_paths = []
    for directory in candidate_dirs:
        for name in candidate_names:
            path = os.path.join(directory, name)
            checked_paths.append(path)
            if os.path.exists(path):
                return path

    checked_text = '\n'.join(f'  - {path}' for path in checked_paths)
    raise FileNotFoundError(f'未找到任何候选文件，请检查数据路径或文件名：\n{checked_text}')

    #数据头修改forecast_v2 / dataset_X
def resolve_x_path(year: int) -> str:
    candidate_names = [
        f'forecast_v2_X_{year}.pt',
        'forecast_v2_X.pt' if year == TRAIN_YEARS[0] else f'forecast_v2_X_{year}.pt',
        # f'dataset_X_{year}.pt',
        # 'dataset_X.pt' if year == TRAIN_YEARS[0] else f'dataset_X_{year}.pt',
    ]
    return find_existing_file(X_CANDIDATE_DIRS, candidate_names)


def resolve_y_path(year: int, label_mode: str) -> str:
    if label_mode == 'kmeans':
        candidate_names = [
            f'forecast_v2_Y_{year}.pt',
            'forecast_v2_Y.pt' if year == TRAIN_YEARS[0] else f'forecast_v2_Y_{year}.pt',
            #  f'dataset_Y_{year}.pt',
            # 'dataset_Y.pt' if year == TRAIN_YEARS[0] else f'dataset_Y_{year}.pt',
        ]
    elif label_mode == 'threshold':
        candidate_names = [
            f'forecast_v2_Y_{year}.pt',
            'forecast_v2_Y.pt',
            # f'dataset_Y_{year}_threshold.pt',
            # 'dataset_Y_threshold.pt',
        ]
    else:
        raise ValueError(f'不支持的 LABEL_MODE: {label_mode}')

    return find_existing_file(Y_CANDIDATE_DIRS, candidate_names)


def build_forecast_dataset(x_tensor: torch.Tensor, y_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if x_tensor.ndim != 5:
        raise ValueError(f'X_tensor 形状应为 (Batch, Time, Channel, H, W)，当前为 {x_tensor.shape}')
    if y_tensor.ndim != 3:
        raise ValueError(f'Y_tensor 形状应为 (Batch, H, W)，当前为 {y_tensor.shape}')
    if x_tensor.shape[0] != y_tensor.shape[0]:
        raise ValueError('X_tensor 与 Y_tensor 的样本数不一致。')
    if x_tensor.shape[1] <= FORECAST_TARGET_MONTH_INDEX:
        raise ValueError(
            f'当前样本时间步数为 {x_tensor.shape[1]}，不足以预测索引 {FORECAST_TARGET_MONTH_INDEX} 对应的未来月份。'
        )
    if FORECAST_INPUT_STEPS >= x_tensor.shape[1]:
        raise ValueError('FORECAST_INPUT_STEPS 必须小于总时间步数，否则不构成预测任务。')

    forecast_x = x_tensor[:, :FORECAST_INPUT_STEPS, :, :, :].contiguous()
    forecast_y = y_tensor.contiguous()
    return forecast_x, forecast_y


def load_year_pair(year: int, label_mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    x_path = resolve_x_path(year)
    y_path = resolve_y_path(year, label_mode)
    print(f'加载 {year} 年特征: {x_path}')
    print(f'加载 {year} 年标签({label_mode}): {y_path}')
    
    # 因为 V2 数据已经切片好了，直接把它当做 forecast_x 和 forecast_y
    forecast_x = torch.load(x_path, map_location='cpu')
    forecast_y = torch.load(y_path, map_location='cpu')

    # -------- 新增：限制最大样本量，防止内存爆炸 --------
    MAX_SAMPLES = 1000  # 你可以改成 1000 或 2000，视你的内存而定
    if forecast_x.shape[0] > MAX_SAMPLES:
        forecast_x = forecast_x[:MAX_SAMPLES]
        forecast_y = forecast_y[:MAX_SAMPLES]
    
    print(f'已加载 V2 预测样本: X={forecast_x.shape}, Y={forecast_y.shape}')
    return forecast_x, forecast_y


def compute_class_weights(y_train: torch.Tensor, device: torch.device) -> torch.Tensor:
    all_labels = y_train.flatten().long()
    class_counts = torch.bincount(all_labels, minlength=4)
    class_counts = torch.clamp(class_counts, min=1)
    class_weights = len(all_labels) / (len(class_counts) * class_counts.float())
    return class_weights.to(device)


def build_model(active_model: str, device: torch.device):
    if active_model in ['convlstm_attn', 'convlstm_no_attn']:
        model_config = copy.deepcopy(model_params['convlstm'])
        model_config['core']['window_in'] = FORECAST_INPUT_STEPS
        use_attention = active_model == 'convlstm_attn'
        if use_attention:
            model_config['core']['input_attn_params']['input_dim'] = FORECAST_INPUT_STEPS
            input_attn_params = model_config['core']['input_attn_params']
        else:
            input_attn_params = None

        model = ConvLSTM(
            input_size=model_config['core']['input_size'],
            window_in=model_config['core']['window_in'],
            num_layers=model_config['core']['num_layers'],
            encoder_params=model_config['core']['encoder_params'],
            input_attn_params=input_attn_params,
            device=device,
        ).to(device)
        model_name = 'convlstm_attn' if use_attention else 'convlstm_no_attn'
    elif active_model == 'convgru':
        model_config = copy.deepcopy(model_params['convgru'])
        model_config['core']['window_in'] = FORECAST_INPUT_STEPS
        model = ConvGRU(
            input_size=model_config['core']['input_size'],
            window_in=model_config['core']['window_in'],
            num_layers=model_config['core']['num_layers'],
            encoder_params=model_config['core']['encoder_params'],
            num_classes=model_config['core']['num_classes'],
            device=device,
        ).to(device)
        model_name = 'convgru'
    elif active_model == 'traj_gru':
        model_config = copy.deepcopy(model_params['traj_gru'])
        model_config['core']['window_in'] = FORECAST_INPUT_STEPS
        model = TrajGRU(
            input_size=model_config['core']['input_size'],
            window_in=model_config['core']['window_in'],
            window_out=model_config['core']['window_out'],
            encoder_params=model_config['core']['encoder_params'],
            decoder_params=model_config['core']['decoder_params'],
            num_classes=model_config['core']['num_classes'],
            device=device,
        ).to(device)
        model_name = 'traj_gru'
    else:
        raise ValueError('当前预测任务支持: convlstm_attn、convlstm_no_attn、convgru、traj_gru。')

    save_name = f'drought_forecast_{model_name}_best_{LABEL_MODE}.pth'
    return model, model_config, save_name, model_name


def main():
    print('正在加载历史数据并构造预测样本...')
    print(f'使用模型: {ACTIVE_MODEL}')
    print(f'标签方案: {LABEL_MODE}')
    print(f'预测设定: 输入前 {FORECAST_INPUT_STEPS} 个月，预测第 {FORECAST_TARGET_MONTH_INDEX + 1} 个月旱情')

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

    actual_channels = x_train.shape[2]
    # 明确只更新我们当前使用的几个模型的通道维度（input_dim），绝不能动空间尺寸（input_size）
    for model_key in ['convlstm', 'convgru', 'traj_gru']: 
        model_params[model_key]['core']['encoder_params']['input_dim'] = actual_channels
    print(f'已自动将模型输入通道数自适应调整为: {actual_channels}')

    print('正在计算损失函数类别权重...')
    class_weights = compute_class_weights(y_train, device)
    print(f'各干旱等级权重: {class_weights.cpu().numpy()}')

    model, model_config, save_name, model_name = build_model(ACTIVE_MODEL, device)
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

    print('开始预测模型训练...')
    losses, best_train, best_val = trainer.train(model=model, batch_generator=batch_generator)

    print(f'\n开始在 {TEST_YEAR} 年未见过的测试集上进行最终预测评估...')
    test_loss, test_metrics = trainer.evaluate(model=model, batch_generator=batch_generator)
    print(f'测试集结果 -> Loss: {test_loss:.5f}, ' + trainer.get_metric_string(test_metrics))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, save_name)
    torch.save(model.state_dict(), save_path)
    print(f'训练完成，最优预测模型 {model_name} 权重已保存至：{save_path}！')
    print(f'本次训练标签方案：{LABEL_MODE}')
    print(f'最佳训练指标：{best_train}')
    print(f'最佳验证指标：{best_val}')
    print(f'测试指标：{test_metrics}')
    print(f'损失曲线记录长度：train={len(losses[0])}, val={len(losses[1])}')


if __name__ == '__main__':
    main()
