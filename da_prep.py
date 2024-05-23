#%%
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

#%% read csv 
df = pd.read_excel('BCH.xlsx')
#%% Filter and select columns
var_col = ['單據日期', '產品編號', '數量']
condition_1 = (df['出貨倉'] == 'W111') | (df['出貨倉'] == 'W113') | (df['出貨倉'] == 'W200')
condition_2 = (df['出貨倉'] == 'W121') | (df['出貨倉'] == 'W123') | (df['出貨倉'] == 'W125')
condition_3 = (df['出貨倉'] == 'W131') | (df['出貨倉'] == 'W132') | (df['出貨倉'] == 'W135')

part1_df = df[condition_1][var_col]
part2_df = df[condition_2][var_col]
part3_df = df[condition_3][var_col]
#%%aggregate_transpose
def aggregate_and_transpose(df):
    # Convert the date column to datetime format
    df['單據日期'] = pd.to_datetime(df['單據日期'])

    # Extract month and year from the date column
    df['月份'] = df['單據日期'].dt.month
    df['年'] = df['單據日期'].dt.year

    # Create a new column combining year and month for easier pivoting
    df['年_月'] = df['年'].astype(str) + '-' + df['月份'].astype(str)

    # Group by '產品編號' and '年_月', then sum the '數量' column
    grouped_df = df.groupby(['產品編號', '年_月'])['數量'].sum().reset_index()

    # Pivot the data to get the desired format
    pivot_df = grouped_df.pivot(index='年_月', columns='產品編號', values='數量').fillna(0)
    # Add a '總計' column that sums the rows
    pivot_df['總計'] = pivot_df.sum(axis=1)

    # Reset the index to make '產品編號' a column again
    pivot_df.reset_index(inplace=True)
     # Sort the pivot table by '年_月' column
    pivot_df = pivot_df.sort_values(by='年_月')

    return pivot_df

# Apply the function to each part
part1_pivot = aggregate_and_transpose(part1_df)
part2_pivot = aggregate_and_transpose(part2_df)
part3_pivot = aggregate_and_transpose(part3_df)

#%%
# Save the transposed data to new CSV files
part1_pivot.to_csv('part1_transposed.csv', index=False)
part2_pivot.to_csv('part2_transposed.csv', index=False)
part3_pivot.to_csv('part3_transposed.csv', index=False)


