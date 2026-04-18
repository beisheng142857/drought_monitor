import ee
import geemap
import time  # 导入时间库用于监控

# 1. 初始化
# ee.Authenticate()
ee.Initialize(project='ee-195273zyk')

# 2. 定义研究区
roi = ee.Geometry.Rectangle([114, 34, 118, 37])

def mask_s2_clouds(image):
    qa = image.select('QA60')
    mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(mask).divide(10000)

target_years = list(range(2020, 2026))
target_months = [4]

print(f"📡 正在向 Google 云端提交任务，请通过下方的状态监控查看进度...\n")

submitted_task_ids = []
skipped_tasks = []

for year in target_years:
    for month in target_months:
        start_date = f"{year}-{month:02d}-01"
        end_date = f"{year + 1}-01-01" if month == 12 else f"{year}-{(month + 1):02d}-01"

        # --- 计算特征 ---
        s2_col = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(roi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
            .map(mask_s2_clouds)
        )
        s2_count = s2_col.size().getInfo()
        if s2_count == 0:
            skipped_tasks.append((year, month, 'S2 empty after filters'))
            print(f"⚠️ 跳过 {year}-{month:02d}: Sentinel-2 为空，无法计算 NDVI")
            continue

        s1_col = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(roi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
        )
        s1_count = s1_col.size().getInfo()
        if s1_count == 0:
            skipped_tasks.append((year, month, 'S1 empty after filters'))
            print(f"⚠️ 跳过 {year}-{month:02d}: Sentinel-1 为空，无法生成 VV/VH")
            continue

        s2_median = s2_col.median()
        ndvi = s2_median.normalizedDifference(['B8', 'B4']).rename('NDVI')

        s1_desc = s1_col.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
        s1_final_col = ee.ImageCollection(ee.Algorithms.If(s1_desc.size().gt(0), s1_desc, s1_col))
        s1_median = s1_final_col.select(['VV', 'VH']).median()

        ratio = s1_median.select('VV').divide(s1_median.select('VH')).rename('VVVH')
        fused = ndvi.addBands([s1_median, ratio]).toFloat().clip(roi)

        # --- 提交到 Drive (这是处理 299MB 数据的唯一稳妥方法) ---
        task_id = f'Fused_100m_{year}_{month:02d}'
        task = ee.batch.Export.image.toDrive(
            image=fused,
            description=task_id,
            folder='GEE_Drought_Project',
            scale=100,
            region=roi,
            maxPixels=1e13
        )
        task.start()
        submitted_task_ids.append(task_id)
        print(f"✅ 任务已提交: {task_id}")

if skipped_tasks:
    print("\n⚠️ 以下月份因数据为空被跳过：")
    for year, month, reason in skipped_tasks:
        print(f"   - {year}-{month:02d}: {reason}")

if not submitted_task_ids:
    print("\n❌ 没有成功提交任何任务，请检查时间范围、ROI 或数据源可用性。")
else:
    # --- 实时进度监控器 ---
    print("\n⏳ 正在监控云端计算进度 (每 30 秒更新一次)...")
    task_states = {task_id: 'SUBMITTED' for task_id in submitted_task_ids}

    while True:
        operations = ee.data.listOperations()
        operations_by_desc = {
            op.get('metadata', {}).get('description'): op.get('metadata', {})
            for op in operations
        }

        completed_or_failed = 0
        for task_id in submitted_task_ids:
            metadata = operations_by_desc.get(task_id, {})
            state = metadata.get('state', task_states.get(task_id, 'UNKNOWN'))
            task_states[task_id] = state
            print(f"   [监控] {task_id}: 当前状态 -> {state}")

            if state in ['COMPLETED', 'FAILED', 'CANCELLED', 'SUCCEEDED']:
                completed_or_failed += 1
                if state == 'FAILED':
                    error_message = metadata.get('error', {}).get('message', 'Unknown error')
                    print(f"      ↳ 失败原因: {error_message}")

        if completed_or_failed == len(submitted_task_ids):
            print("\n🎉 所有已提交任务均已结束，请前往 Google Drive 和任务面板核查结果。")
            break

        time.sleep(30)  # 每 30 秒检查一次，避免刷屏

