import ee
import geemap
import time  # 导入时间库用于监控

# 1. 初始化
ee.Authenticate()
ee.Initialize(project='ee-195273zyk')

# 2. 定义研究区
roi = ee.Geometry.Rectangle([114, 34, 118, 37])

def mask_s2_clouds(image):
    qa = image.select('QA60')
    mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(mask).divide(10000)

target_year = 2021
target_months = [5, 6, 7, 8, 9]

print(f"📡 正在向 Google 云端提交任务，请通过下方的状态监控查看进度...\n")

for month in target_months:
    start_date = f"{target_year}-{month:02d}-01"
    end_date = f"{target_year + 1}-01-01" if month == 12 else f"{target_year}-{(month + 1):02d}-01"

    # --- 计算特征 ---
    s2_median = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(roi).filterDate(start_date, end_date).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).map(mask_s2_clouds).median()
    ndvi = s2_median.normalizedDifference(['B8', 'B4']).rename('NDVI')

    s1_col = ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(roi).filterDate(start_date, end_date).filter(ee.Filter.eq('instrumentMode', 'IW'))
    s1_desc = s1_col.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
    s1_final_col = ee.Algorithms.If(s1_desc.size().gt(0), s1_desc, s1_col)
    s1_median = ee.ImageCollection(s1_final_col).select(['VV', 'VH']).median()

    ratio = s1_median.select('VV').divide(s1_median.select('VH')).rename('VVVH')
    fused = ndvi.addBands([s1_median, ratio]).toFloat().clip(roi)

    # --- 提交到 Drive (这是处理 299MB 数据的唯一稳妥方法) ---
    task_id = f'Fused_100m_{target_year}_{month:02d}'
    task = ee.batch.Export.image.toDrive(
        image=fused,
        description=task_id,
        folder='GEE_Drought_Project',
        scale=100,
        region=roi.getInfo()['coordinates'],
        maxPixels=1e13
    )
    task.start()
    print(f"✅ 任务已提交: {task_id}")

# --- 核心新增：实时进度监控器 ---
print("\n⏳ 正在监控云端计算进度 (每 30 秒更新一次)...")
while True:
    tasks = ee.data.listOperations() # 获取所有任务状态
    active_tasks = [t for t in tasks if t['metadata']['state'] in ['READY', 'RUNNING']]

    if not active_tasks:
        print("\n🎉 所有月份的任务均已计算完成！请前往 Google Drive 查收。")
        break

    for t in active_tasks:
        desc = t['metadata']['description']
        state = t['metadata']['state']
        print(f"   [监控] {desc}: 当前状态 -> {state}")

    time.sleep(30) # 每 30 秒检查一次，避免刷屏
