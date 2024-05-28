#%%
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

#%% read csv 
df = pd.read_excel('sale_data.xlsx')
print(df.head())
#%% Filter and select columns
var_col = ['單據日期', '產品編號', '數量','單價']
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
    df['月份'] = df['單據日期'].dt.strftime("%m")
    df['年'] = df['單據日期'].dt.strftime("%Y")
    df=df[~df['產品編號'].str.startswith('0')]
    # Create a new column combining year and month for easier pivoting
    df['年_月'] = df['年'].astype(str) + '-' + df['月份'].astype(str)
    # Group by '產品編號' and '年_月', then sum the '數量' column
    grouped_df = df.groupby(['產品編號','年_月'])['數量'].sum().reset_index()
    
    # Pivot the data to get the desired format
    pivot_df = grouped_df.pivot(index='產品編號', columns='年_月', values='數量').fillna(0)
    # Add a '總計' column that sums the rows
    pivot_df['總計'] = pivot_df.sum(axis=1)

    # Reset the index to make '產品編號' a column again
    pivot_df.reset_index(inplace=True)
    # Sort the pivot table by '年_月' column
    #pivot_df = pivot_df.sort_values(by='年_月')

    return pivot_df
# Apply the function to each part
part1_pivot = aggregate_and_transpose(part1_df)
part2_pivot = aggregate_and_transpose(part2_df)
part3_pivot = aggregate_and_transpose(part3_df)


# %% calculate_abc_classifiaction
def calculate_abc_classification(df):
    # Sort the DataFrame by '總計' in descending order
    df= df.sort_values(by='總計', ascending=False)
    
    # Calculate cumulative sum of '總計'
    df['總計_cum'] = df['總計'].cumsum()
    
    # Calculate the total sum of '總計'
    df['總計_all'] = df['總計'].sum()
    
    # Calculate the cumulative percentage of '總計'
    df['sku_總計_percent'] = df['總計_cum'] / df['總計_all']
    
    # Define the ABC classification function
    def condition_abc(x):
        if x > 0 and x < 0.80:
            return 'A'
        elif x >= 0.80 and x < 0.90:
            return 'B'
        else:
            return 'C'
    
    # Apply the ABC classification to the DataFrame
    df['ABC'] = df['sku_總計_percent'].apply(condition_abc)
    return df
#%%
def apply_abc_classification(part1_pivot, part2_pivot, part3_pivot):
    part1_pivot = calculate_abc_classification(part1_pivot)
    part2_pivot = calculate_abc_classification(part2_pivot)
    part3_pivot = calculate_abc_classification(part3_pivot)
    return part1_pivot, part2_pivot, part3_pivot
# Assuming part1_pivot, part2_pivot, and part3_pivot are predefined DataFrames
part1_pivot, part2_pivot, part3_pivot = apply_abc_classification(part1_pivot, part2_pivot, part3_pivot)
# %%filter out the data if the data cols coantin the "A"
part1_pivot=part1_pivot[part1_pivot['ABC'].str.contains('A')]
part2_pivot=part2_pivot[part2_pivot['ABC'].str.contains('A')]
part3_pivot=part3_pivot[part3_pivot['ABC'].str.contains('A')]
# %%
# %%

# %%
