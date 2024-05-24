#%%
import pandas as pd
import numpy as np
import pmdarima as pm
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
#%%
# 給定2023年的每月數據
data = [ 0, 0, 439, 75, 64, 224,200,128,0,256,0,884]





#%%
# 創建pandas系列
date_range = pd.date_range(start='2023-1', periods=len(data), freq='M')
series = pd.Series(data, index=date_range)
print(series)
#%%
# 進行ADF檢驗
adf_result = adfuller(series)
print("原始數據的ADF檢驗結果:", adf_result)
#%%
# 使用auto_arima自動選擇最佳的ARIMA模型參數
model = pm.auto_arima(series, start_p=0, start_q=0,
                      information_criterion='aic',
                      test='adf',       # 使用ADF檢驗來確定最佳的'd'
                      max_p=8, max_q=8, # p和q的最大值
                      m=1,              # 數據的頻率
                      d=None,           # 讓模型自動選擇'd'
                      seasonal=False,   # 非季節性
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model.summary())
#%%
# 預測下一個月（2024-01）
forecast = model.predict(n_periods=1)
forecast_value = forecast[0]
print(f"2024 年 1 月的預測數據為: {forecast_value}")

# %%
# 如果預測結果為負數，則調整為0
if forecast_value < 0:
    forecast_value = 0

print(f"2024 年 1 月的預測數據為: {forecast_value}")

# %%
# 繪製原始數據和預測值
plt.figure(figsize=(12, 6))
plt.plot(series, marker='o', label='原始數據')
plt.plot(pd.date_range(start='2024-01', periods=1, freq='M'), [forecast_value], marker='o', color='red', label='預測數據')
plt.title('原始數據與2024年1月的預測數據')
plt.xlabel('日期')
plt.ylabel('數值')
plt.legend()
plt.grid(True)
plt.show()

# %%
