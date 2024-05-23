import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt

# 給定2023年的每月數據
data = [0, 0, 439, 75, 64, 224, 200, 128, 0, 256, 0, 884]

# 創建pandas DataFrame，Prophet要求的格式
date_range = pd.date_range(start='2023-01', periods=len(data), freq='M')
df = pd.DataFrame({'ds': date_range, 'y': data})

# 訓練Prophet模型
model = Prophet()
model.fit(df)

# 進行未來1個月的預測
future = model.make_future_dataframe(periods=1, freq='M')
forecast = model.predict(future)

# 獲取預測值
forecast_value = forecast.iloc[-1]['yhat']
print(f"2024 年 1 月的預測數據為: {forecast_value}")

# 如果預測結果為負數，則調整為0
if forecast_value < 0:
    forecast_value = 0

print(f"2024 年 1 月的預測數據為: {forecast_value}")

# 繪製預測結果
fig = model.plot(forecast)
plt.title('原始數據與2024年1月的預測數據')
plt.xlabel('日期')
plt.ylabel('數值')
plt.grid(True)
plt.show()

# 繪製原始數據和預測值
plt.figure(figsize=(12, 6))
plt.plot(df['ds'], df['y'], marker='o', label='原始數據')
plt.plot(forecast['ds'], forecast['yhat'], marker='o', color='red', label='預測數據')
plt.title('原始數據與2024年1月的預測數據')
plt.xlabel('日期')
plt.ylabel('數值')
plt.legend()
plt.grid(True)
plt.show()
