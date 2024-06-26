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
import pmdarima as pm


# %% create dataframes 
# 生成不平稳的时间序列
np.random.seed(0)
n = 100
x = np.cumsum(np.random.randn(n))

# 把它转换成Pandas的DataFrame格式
df = pd.DataFrame(x, columns=['value'])
print(df)
# %%
plt.hist(df['value'])
# %%
plt.plot(df['value'])
# %% 數據平穩化
df['diff_1']=df['value'].diff(1)
df['diff_2']=df['value'].diff(1).diff(1)
#%%分別劃出ACF(自相關)和PACF(偏自相關)
#%%繪製圖
fig,axes=plt.subplots(3,3,figsize=(12,6))
axes[0,0].plot(df['value']); axes[0,0].set_title('original series')
plot_acf(df['value'],ax=axes[0,1])
plot_pacf(df['value'],ax=axes[0,2])
axes[1,0].plot(df['diff_1']); axes[1,0].set_title('1st order differencing')
plot_acf(df['diff_1'].dropna(),ax=axes[1,1])
plot_pacf(df['diff_1'].dropna(),ax=axes[1,2])
axes[2,0].plot(df['diff_2']); axes[2,0].set_title('2st order differencing')
plot_acf(df['diff_2'].dropna(),ax=axes[2,1])
plot_pacf(df['diff_2'].dropna(),ax=axes[2,2])


plt.tight_layout()
plt.show()

# %%數據平穩檢驗
print('原始序列結果檢驗為:',ADF(df['value']))
#依次為adf,pvalue.... < p-value<0.05
print('一次序列結果檢驗為:',ADF(df['diff_1'][1:].dropna()))
print('二次序列結果檢驗為:',ADF(df['diff_2'][1:].dropna()))

#%%目前找到最好(p,d,q)=(0,1,0)
print(df.head())
# %%根據bic or aic 指定 p ,q 
pmax=5
qmax=5
bic_matrix=[]
for p in range(qmax+1):
    tmp=[]
    for q in range(qmax+1):
        try:
            tmp.append(ARIMA(df['value'],order=(p,1,q)).fit().bic)
        except:
            tmp.append(None)
    bic_matrix.append(tmp)
bic_matrix =pd.DataFrame(bic_matrix)    
print(bic_matrix)
p,q=bic_matrix.stack().idxmin()
print(p,q)

#%%根據aic 指定p,q
pmax=5
qmax=5
aic_matrix=[]
for p in range(pmax+1):
    tmp=[]
    for q in range(qmax+1):
        try:
            tmp.append(ARIMA(df['value'],order=(p,1,q)).fit().bic)
        except:
            tmp.append(None)
    aic_matrix.append(tmp)
aic_matrix=pd.DataFrame(aic_matrix)
p,q=aic_matrix.stack().idxmin()
print(p,q)


# %%模型擬合
arima_0_1_0=ARIMA(df['value'],order=(0,1,0)).fit()
print(arima_0_1_0.summary())

# %% 模型檢驗
resid=arima_0_1_0.resid
plt.figure(figsize=(12,8))
plt.plot(resid)



# %%正泰殘差檢檢驗
stats.probplot(resid,dist='norm',plot=plt)
plt.show()
plt.hist(resid,bins=50)
plt.show()

# %%殘差序列自相關 (殘差序列是否獨立)
from statsmodels.stats.stattools import durbin_watson
durbin_watson(arima_0_1_0.resid.values)
#dw檢驗 接近2--正常; 靠近0--正自相關; 靠近4-- 負自相關


# %% apply auto ariam() for single time series 

model = pm.auto_arima(df.value, start_p=1, start_q=1,
                      information_criterion='aic',
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=5, max_q=5, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model.summary())


# %% mutiple time series with auto arima()
import pandas as pd 
def print_model_summaries(df, n_periods=1, seasonal=False):  # Set seasonal to False
    # Iterate over columns in data
    for group in df.columns:
        if group == 'Date':
            continue
        # Fit an ARIMA model using the auto_arima function
        data_actual = df[group].values
        model = pm.auto_arima(data_actual, 
                                  start_p=0, start_q=0,
                                  max_p=12, max_q=12, # maximum p and q
                                  information_criterion='aic',
                                  test='adf',         # use adftest to find optimal 'd'
                                  seasonal=seasonal,  # Set seasonal parameter
                                  m=1,               # frequency of series
                                  d=None,             # let model determine 'd'
                                  D=None,             # let model determine 'D'
                                  trace=True,         # To show the upper part of the model 
                                  error_action='ignore',  
                                  suppress_warnings=True, 
                                  stepwise=True)
        print(f"\nModel summary for {group}:")
        print(model.summary())
        print("\n")

#%%
import pandas as pd
import numpy as np

# Generate a date range for 12 months
dates = pd.date_range(start='2023-01', periods=12, freq='MS')

# Seed for reproducibility
np.random.seed(42)

# Generate random sales data for three products
data = {
    'Date': dates,
    'Product_1': np.random.randint(100, 500, size=12),
    'Product_2': np.random.randint(200, 600, size=12),
    'Product_3': np.random.randint(300, 700, size=12)
}

# Create the DataFrame
df = pd.DataFrame(data)

print(df)

#%%
print_model_summaries(df)

