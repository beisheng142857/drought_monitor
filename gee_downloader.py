import argparse
import time
from typing import Dict

import ee


ee.Initialize(project='ee-195273zyk')

ROI = ee.Geometry.Rectangle([114, 34, 118, 37])
DEFAULT_YEARS = list(range(2021, 2026))
DEFAULT_MONTHS = [4, 5, 6, 7, 8, 9]
DEFAULT_FOLDER = 'GEE_Drought_Project'
DEFAULT_SCALE = 100

S2_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
S1_BANDS = ['VV', 'VH']
DEFAULT_FEATURES = [
    'NDVI', 'EVI', 'NDMI', 'NDWI', 'MSAVI',
    'VV', 'VH', 'VVVH', 'VVDIFFVH', 'RVI'
]


def mask_s2_clouds(image: ee.Image) -> ee.Image:
    qa = image.select('QA60')
    mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(mask).divide(10000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='批量下载 Sentinel-1/Sentinel-2 月尺度特征影像')
    parser.add_argument('--years', nargs='+', type=int, default=DEFAULT_YEARS)
    parser.add_argument('--months', nargs='+', type=int, default=DEFAULT_MONTHS)
    parser.add_argument('--folder', type=str, default=DEFAULT_FOLDER)
    parser.add_argument('--scale', type=int, default=DEFAULT_SCALE)
    parser.add_argument('--s2_cloud_pct', type=float, default=20.0)
    parser.add_argument('--features', nargs='+', default=DEFAULT_FEATURES)
    return parser.parse_args()


def build_s2_feature_dict(s2_image: ee.Image) -> Dict[str, ee.Image]:
    ndvi = s2_image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndmi = s2_image.normalizedDifference(['B8', 'B11']).rename('NDMI')
    ndwi = s2_image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    evi = s2_image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
        {
            'NIR': s2_image.select('B8'),
            'RED': s2_image.select('B4'),
            'BLUE': s2_image.select('B2'),
        },
    ).rename('EVI')
    msavi = s2_image.expression(
        '(2 * NIR + 1 - sqrt((2 * NIR + 1) ** 2 - 8 * (NIR - RED))) / 2',
        {
            'NIR': s2_image.select('B8'),
            'RED': s2_image.select('B4'),
        },
    ).rename('MSAVI')
    return {
        'NDVI': ndvi,
        'EVI': evi,
        'NDMI': ndmi,
        'NDWI': ndwi,
        'MSAVI': msavi,
    }


def build_s1_feature_dict(s1_image: ee.Image) -> Dict[str, ee.Image]:
    vv = s1_image.select('VV').rename('VV')
    vh = s1_image.select('VH').rename('VH')
    safe_vh = vh.where(vh.abs().lt(1e-6), 1e-6)
    vvvh = vv.divide(safe_vh).rename('VVVH')
    vvdiffvh = vv.subtract(vh).rename('VVDIFFVH')
    rvi = s1_image.expression('4 * VH / (VV + VH)', {'VV': vv, 'VH': safe_vh}).rename('RVI')
    return {
        'VV': vv,
        'VH': vh,
        'VVVH': vvvh,
        'VVDIFFVH': vvdiffvh,
        'RVI': rvi,
    }


def build_monthly_feature_image(year: int, month: int, s2_cloud_pct: float, selected_features: list[str]) -> ee.Image:
    start_date = f'{year}-{month:02d}-01'
    end_date = f'{year + 1}-01-01' if month == 12 else f'{year}-{month + 1:02d}-01'

    s2_col = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(ROI)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', s2_cloud_pct))
        .map(mask_s2_clouds)
        .select(S2_BANDS)
    )
    if s2_col.size().getInfo() == 0:
        raise ValueError('Sentinel-2 为空，无法计算光学指数。')

    s1_col = (
        ee.ImageCollection('COPERNICUS/S1_GRD')
        .filterBounds(ROI)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .select(S1_BANDS)
    )
    if s1_col.size().getInfo() == 0:
        raise ValueError('Sentinel-1 为空，无法计算雷达特征。')

    s2_image = s2_col.median()
    s1_desc = s1_col.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
    s1_image = ee.ImageCollection(ee.Algorithms.If(s1_desc.size().gt(0), s1_desc, s1_col)).median()

    feature_dict: Dict[str, ee.Image] = {}
    feature_dict.update(build_s2_feature_dict(s2_image))
    feature_dict.update(build_s1_feature_dict(s1_image))

    unknown_features = [name for name in selected_features if name not in feature_dict]
    if unknown_features:
        raise ValueError(f'不支持的特征名: {unknown_features}')

    ordered_images = [feature_dict[name] for name in selected_features]
    return ee.Image.cat(ordered_images).toFloat().clip(ROI)


def main():
    args = parse_args()
    selected_features = list(dict.fromkeys(args.features))
    submitted_task_ids = []
    skipped_tasks = []

    print('开始向 Earth Engine 提交月尺度导出任务。')
    print(f'年份: {args.years}')
    print(f'月份: {args.months}')
    print(f'导出特征: {selected_features}')

    for year in args.years:
        for month in args.months:
            try:
                image = build_monthly_feature_image(year, month, args.s2_cloud_pct, selected_features)
            except Exception as exc:
                skipped_tasks.append((year, month, str(exc)))
                print(f'跳过 {year}-{month:02d}: {exc}')
                continue

            task_id = f'Fused_100m_{year}_{month:02d}'
            task = ee.batch.Export.image.toDrive(
                image=image,
                description=task_id,
                folder=args.folder,
                scale=args.scale,
                region=ROI,
                maxPixels=1e13,
            )
            task.start()
            submitted_task_ids.append(task_id)
            print(f'已提交任务: {task_id}')

    if skipped_tasks:
        print('\n以下月份被跳过:')
        for year, month, reason in skipped_tasks:
            print(f'  - {year}-{month:02d}: {reason}')

    if not submitted_task_ids:
        print('没有成功提交任何任务，请检查时间范围、ROI 与数据源可用性。')
        return

    print('\n开始监控任务状态，每 30 秒更新一次。')
    task_states = {task_id: 'SUBMITTED' for task_id in submitted_task_ids}
    while True:
        operations = ee.data.listOperations()
        operations_by_desc = {
            op.get('metadata', {}).get('description'): op.get('metadata', {})
            for op in operations
        }

        finished_count = 0
        for task_id in submitted_task_ids:
            metadata = operations_by_desc.get(task_id, {})
            state = metadata.get('state', task_states.get(task_id, 'UNKNOWN'))
            task_states[task_id] = state
            print(f'[监控] {task_id}: {state}')
            if state in ['COMPLETED', 'FAILED', 'CANCELLED', 'SUCCEEDED']:
                finished_count += 1
                if state == 'FAILED':
                    error_message = metadata.get('error', {}).get('message', 'Unknown error')
                    print(f'  失败原因: {error_message}')

        if finished_count == len(submitted_task_ids):
            print('所有已提交任务均已结束，请前往 Drive 核查导出结果。')
            break

        time.sleep(30)


if __name__ == '__main__':
    main()
