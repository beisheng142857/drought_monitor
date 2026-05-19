import argparse
import copy
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

NEW_CODE_DIR = '/root/autodl-tmp/zyk_drought_monitor/proposed_attention_optimization'
ROOT_DIR = '/root/autodl-tmp/zyk_drought_monitor'
if NEW_CODE_DIR not in sys.path:
    sys.path.insert(0, NEW_CODE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from configs.config import model_params
from models.baseline.convgru import ConvGRU
from models.baseline.convlstm import ConvLSTM
from models.baseline.traj_gru import TrajGRU

CLASS_NAMES = ['无旱', '轻旱', '中旱', '重/特旱']
CLASS_COLORMAP = {
    0: (44, 162, 95, 255),
    1: (254, 224, 139, 255),
    2: (244, 109, 67, 255),
    3: (165, 0, 38, 255),
    255: (0, 0, 0, 0),
}
FORECAST_INPUT_STEPS = 4
DEFAULT_INPUT_DIR = '/root/autodl-tmp/data_forecast_V2'
DEFAULT_OUTPUT_DIR = '/root/autodl-tmp/zyk_drought_monitor/drought_outputs/prediction_tifs'
DEFAULT_DATA_DIRS = [
    '/root/autodl-tmp/zyk_drought_monitor/data_V2',
    '/root/autodl-tmp/data_proc',
    '/root/autodl-tmp/data_proc/data_proc',
]
DEFAULT_MONTHS = [4, 5, 6, 7, 8, 9]


@dataclass
class SampleWindow:
    index: int
    window: Window
    transform: rasterio.Affine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='将旱情预测结果导出为带地理参考的 GeoTIFF')
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--sample_index', type=int, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--label_mode', type=str, default='threshold', choices=['threshold', 'kmeans'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--input_dir', type=str, default=DEFAULT_INPUT_DIR)
    parser.add_argument('--data_dirs', nargs='+', default=DEFAULT_DATA_DIRS)
    parser.add_argument('--months', nargs='+', type=int, default=DEFAULT_MONTHS)
    parser.add_argument('--file_prefix', type=str, default='Fused_100m')
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--stride', type=int, default=64)
    parser.add_argument('--nodata_threshold', type=float, default=0.4)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--export_gt', action='store_true')
    parser.add_argument('--export_reference_band', action='store_true')
    return parser.parse_args()


def build_tiff_paths(year: int, months: Sequence[int], input_dir: str, file_prefix: str) -> List[str]:
    return [os.path.join(input_dir, f'{file_prefix}_{year}_{month:02d}.tif') for month in months]


def validate_paths(paths: Sequence[str]) -> None:
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError('缺少输入文件:\n' + '\n'.join(f'  - {path}' for path in missing))


def find_existing_file(candidate_dirs: Sequence[str], candidate_names: Sequence[str]) -> str:
    for directory in candidate_dirs:
        for name in candidate_names:
            path = os.path.join(directory, name)
            if os.path.exists(path):
                return path
    raise FileNotFoundError(f'未找到候选文件: {candidate_names}')


def resolve_forecast_paths(data_dirs: Sequence[str], label_mode: str, year: int) -> Tuple[str, str]:
    x_path = find_existing_file(data_dirs, [f'forecast_v2_X_{year}.pt'])
    y_names = [f'forecast_v2_Y_{year}.pt', 'forecast_v2_Y.pt'] if label_mode == 'threshold' else [f'forecast_v2_Y_{year}.pt', 'forecast_v2_Y.pt']
    y_path = find_existing_file(data_dirs, y_names)
    return x_path, y_path


def read_window_with_mask(dataset: rasterio.io.DatasetReader, window: Window) -> Tuple[np.ndarray, np.ndarray]:
    data = dataset.read(window=window).astype(np.float32)
    invalid_mask = ~np.isfinite(data)
    if dataset.nodata is not None:
        invalid_mask |= data == dataset.nodata
    return data, invalid_mask


def enumerate_valid_windows(
    tiff_paths: Sequence[str],
    patch_size: int,
    stride: int,
    nodata_threshold: float,
) -> Tuple[List[SampleWindow], dict]:
    validate_paths(tiff_paths)
    datasets = [rasterio.open(path) for path in tiff_paths]
    try:
        ref = datasets[0]
        meta = ref.meta.copy()
        height, width, count = ref.height, ref.width, ref.count

        for dataset in datasets[1:]:
            if dataset.height != height or dataset.width != width or dataset.count != count:
                raise ValueError('输入 TIFF 的空间尺寸或波段数不一致。')

        windows: List[SampleWindow] = []
        sample_index = 0
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                window = Window(x, y, patch_size, patch_size)
                invalid_masks = []
                for dataset in datasets:
                    _, invalid_mask = read_window_with_mask(dataset, window)
                    invalid_masks.append(invalid_mask)
                invalid_ratio = np.stack(invalid_masks, axis=0).mean()
                if invalid_ratio > nodata_threshold:
                    continue
                windows.append(
                    SampleWindow(
                        index=sample_index,
                        window=window,
                        transform=rasterio.windows.transform(window, ref.transform),
                    )
                )
                sample_index += 1
        return windows, meta
    finally:
        for dataset in datasets:
            dataset.close()


def infer_model_type_from_ckpt(ckpt_path: str) -> str:
    lower = os.path.basename(ckpt_path).lower()
    if 'forecast_convlstm_attn' in lower:
        return 'convlstm_attn'
    if 'forecast_convlstm_no_attn' in lower:
        return 'convlstm_no_attn'
    if 'forecast_convgru' in lower:
        return 'convgru'
    if 'forecast_traj_gru' in lower or 'forecast_trajgru' in lower:
        return 'traj_gru'
    raise ValueError(f'无法从文件名推断预测模型类型: {ckpt_path}')


def build_model(model_type: str, device: torch.device, actual_channels: int) -> torch.nn.Module:
    if model_type in ['convlstm_attn', 'convlstm_no_attn']:
        cfg = copy.deepcopy(model_params['convlstm']['core'])
        cfg['window_in'] = FORECAST_INPUT_STEPS
        cfg['encoder_params']['input_dim'] = actual_channels
        attn = copy.deepcopy(cfg['input_attn_params']) if model_type == 'convlstm_attn' else None
        if attn is not None:
            attn['input_dim'] = actual_channels
        model = ConvLSTM(cfg['input_size'], cfg['window_in'], cfg['num_layers'], cfg['encoder_params'], attn, device)
    elif model_type == 'convgru':
        cfg = copy.deepcopy(model_params['convgru']['core'])
        cfg['window_in'] = FORECAST_INPUT_STEPS
        cfg['encoder_params']['input_dim'] = actual_channels
        model = ConvGRU(cfg['input_size'], cfg['window_in'], cfg['num_layers'], cfg['encoder_params'], device, cfg['num_classes'])
    else:
        cfg = copy.deepcopy(model_params['traj_gru']['core'])
        cfg['window_in'] = FORECAST_INPUT_STEPS
        cfg['encoder_params']['input_dim'] = actual_channels
        model = TrajGRU(cfg['input_size'], cfg['window_in'], cfg['window_out'], cfg['encoder_params'], cfg['decoder_params'], device, cfg['num_classes'])
    return model.to(device)


def load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location='cpu')
    if isinstance(obj, dict) and 'state_dict' in obj and isinstance(obj['state_dict'], dict):
        return obj['state_dict']
    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj
    raise ValueError(f'无法识别 checkpoint 格式: {path}')


def load_model_from_checkpoint(ckpt_path: str, device: torch.device, actual_channels: int):
    model_type = infer_model_type_from_ckpt(ckpt_path)
    model = build_model(model_type, device, actual_channels)
    model.load_state_dict(load_state_dict(ckpt_path), strict=False)
    model.eval()
    return model, model_type


def build_forecast_dataset(x_tensor: torch.Tensor, y_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return x_tensor[:, :FORECAST_INPUT_STEPS].contiguous(), y_tensor.contiguous()


def predict_single(model: torch.nn.Module, x_sample: torch.Tensor, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        x = x_sample.unsqueeze(0).float().to(device)
        hidden = model.init_hidden(batch_size=1) if hasattr(model, 'init_hidden') else None
        logits = model(x=x, hidden=hidden)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred


def write_single_band_tif(
    output_path: str,
    array: np.ndarray,
    meta: dict,
    transform: rasterio.Affine,
    nodata: int | float,
    dtype: str,
    band_name: str,
    use_colormap: bool,
) -> None:
    profile = meta.copy()
    profile.update({
        'driver': 'GTiff',
        'height': array.shape[0],
        'width': array.shape[1],
        'count': 1,
        'dtype': dtype,
        'transform': transform,
        'compress': 'lzw',
        'nodata': nodata,
    })
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(array.astype(dtype), 1)
        dst.set_band_description(1, band_name)
        if use_colormap:
            dst.write_colormap(1, CLASS_COLORMAP)
            dst.update_tags(classes=';'.join(f'{i}:{name}' for i, name in enumerate(CLASS_NAMES)))


def main() -> None:
    args = parse_args()
    validate_paths([args.checkpoint])
    os.makedirs(args.output_dir, exist_ok=True)

    tiff_paths = build_tiff_paths(args.year, args.months, args.input_dir, args.file_prefix)
    valid_windows, raster_meta = enumerate_valid_windows(
        tiff_paths=tiff_paths,
        patch_size=args.patch_size,
        stride=args.stride,
        nodata_threshold=args.nodata_threshold,
    )
    if args.sample_index < 0 or args.sample_index >= len(valid_windows):
        raise IndexError(f'sample_index 越界，当前有效样本数为 {len(valid_windows)}')

    x_path, y_path = resolve_forecast_paths(args.data_dirs, args.label_mode, args.year)
    x_all_raw = torch.load(x_path, map_location='cpu')
    y_all_raw = torch.load(y_path, map_location='cpu')
    x_all, y_all = build_forecast_dataset(x_all_raw, y_all_raw)
    if args.sample_index >= x_all.shape[0]:
        raise IndexError(f'sample_index 超出张量样本数，当前张量样本数为 {x_all.shape[0]}')

    x_sample = x_all[args.sample_index]
    y_true = y_all[args.sample_index].cpu().numpy().astype(np.uint8)
    actual_channels = x_all.shape[2]
    model, model_type = load_model_from_checkpoint(args.checkpoint, torch.device(args.device), actual_channels)
    y_pred = predict_single(model, x_sample, torch.device(args.device))

    sample_window = valid_windows[args.sample_index]
    base_name = f'{model_type}_year{args.year}_sample{args.sample_index:04d}'

    pred_path = os.path.join(args.output_dir, f'{base_name}_pred.tif')
    write_single_band_tif(
        output_path=pred_path,
        array=y_pred,
        meta=raster_meta,
        transform=sample_window.transform,
        nodata=255,
        dtype='uint8',
        band_name='predicted_drought_class',
        use_colormap=True,
    )
    print(f'[OK] 已导出预测 GeoTIFF: {pred_path}')

    if args.export_gt:
        gt_path = os.path.join(args.output_dir, f'{base_name}_gt.tif')
        write_single_band_tif(
            output_path=gt_path,
            array=y_true,
            meta=raster_meta,
            transform=sample_window.transform,
            nodata=255,
            dtype='uint8',
            band_name='ground_truth_drought_class',
            use_colormap=True,
        )
        print(f'[OK] 已导出真值 GeoTIFF: {gt_path}')

    if args.export_reference_band:
        ref_band = x_sample[-1, 0].cpu().numpy().astype(np.float32)
        ref_path = os.path.join(args.output_dir, f'{base_name}_reference_ndvi.tif')
        write_single_band_tif(
            output_path=ref_path,
            array=ref_band,
            meta=raster_meta,
            transform=sample_window.transform,
            nodata=np.nan,
            dtype='float32',
            band_name='reference_ndvi_last_input_step',
            use_colormap=False,
        )
        print(f'[OK] 已导出参考底图 GeoTIFF: {ref_path}')


if __name__ == '__main__':
    main()
