#%% install packages 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from statsmodels.tsa.arima.model import ARIMA 
import statsmodels as sm 
from scipy import stats 
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller as ADF


# %% create dataframes 
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
data = np.random.poisson(lam=20, size=len(dates))
# Create a DataFrame
df = pd.DataFrame(data, index=dates, columns=['value'])
# %%
plt.hist(df)
# %%
plt.plot(df)
# %% 數據平穩化
df['diff_1']=df['value'].diff(1)
df['diff_2']=df['value'].diff(1).diff(1)
#%%分別劃出ACF(自相關)和PACF(偏自相關)
#%%繪製圖
fig,axes=plt.subplots(3,2,figsize=(10,6))
axes[0,0].plot(df['value']); axes[0,0].set_title('original series')
plot_acf(df['value'],ax=axes[0,1])
axes[1,0].plot(df['diff_1']); axes[1,0].set_title('1st order differencing')
plot_acf(df['diff_1'].dropna(),ax=axes[1,1])
axes[2,0].plot(df['diff_2']); axes[2,0].set_title('2st order differencing')
plot_acf(df['diff_2'].dropna(),ax=axes[2,1])
plt.tight_layout()
plt.show()

# %%數據平穩檢驗
print('原始序列結果檢驗為:',ADF(df['value']))
#依次為adf,pvalue.... < p-value<0.05
print('一次序列結果檢驗為:',ADF(df['diff_1'][1:].dropna()))
print('二次序列結果檢驗為:',ADF(df['diff_2'][1:].dropna()))


# %%
