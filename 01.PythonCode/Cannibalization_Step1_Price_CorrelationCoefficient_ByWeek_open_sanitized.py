# -*- coding: utf-8 -*-
# === Injected config shim for sanitized version ===
import os, os.path as _osp

# DB schema/table indirection (medium risk → parameterized)
SCHEMA_PUBLIC = os.getenv("DB_SCHEMA_PUBLIC", "public")
TABLE_TRANSACTION_ITEMS = os.getenv("TABLE_TRANSACTION_ITEMS", "transaction_items_partitioned")
TABLE_RELATIONSHIPS = os.getenv("TABLE_RELATIONSHIPS", "relationships")
TABLE_THREAD_DEAL = os.getenv("TABLE_THREAD_DEAL", "thread_deal")

# Business group UUID (medium risk → parameterized)
SHOP_GROUP_UUID = os.getenv("SHOP_GROUP_UUID", "{SHOP_GROUP_UUID}")

# Paths (medium risk → parameterized)
DATA_DIR = os.getenv("DATA_DIR", "data")
INTERNAL_SHARE_PATH = os.getenv("INTERNAL_SHARE_PATH", "data/share")
REF_STORE_CSV = os.getenv("REF_STORE_CSV", _osp.join(DATA_DIR, "Ref_StoreNameCode.csv"))
\1

# === Business-Logic Strategy Layer (sanitized) ===
# This repository ships with *placeholder* logic so you can run the pipeline end-to-end
# without revealing proprietary modelling decisions. Override by providing your own
# module on PYTHONPATH named `pricing_strategy.py` that defines the same functions.

try:
    import pricing_strategy as _strategy
except Exception:  # pragma: no cover - fall back to public-safe defaults
    class _strategy:
        @staticmethod
        def select_cannibal_pairs(df_weekly, **kwargs):
    return _strategy.select_cannibal_pairs(df_weekly, **kwargs)

Created on Wed Jul  3 11:40:10 2024

@author: danielp
"""

import pandas as pd
import datetime as dt
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define a function to replace invalid characters
import re
def sanitize_filename(name):
    return re.sub(r'[\/:*?"<>|]', '_', name)




############### Block1: Read raw data ###############

### Step 1: Read dataset
# File path
fld_data = r"os.path.join(DATA_DIR, '')"
file_wk_sales = fld_data + "WeeklyQTY_SelectedCategories_24Q1_+11.csv"
print(file_wk_sales)

# Read file
df_raw_all = pd.read_csv(file_wk_sales, header=0)
df_raw_all.dtypes

############### End of Block1: Read raw data ###############





############### Block2: Process Dataframe- Combine category ###############

##### Step1: Filter categories & regroup categories ##########
# Filter wanted categories
list_space = ['BASIC FOOD PRODUCTS'
                ,'BEVERAGES'
                ,'EDIBLE'
                ,'BEER - PACKS'
                ,'WINE - RED'
                ,'WINE - WHITE'
                ,'SNACKS AND SAVOURIES'
                ]

# Filter
df_raw_all = df_raw_all[df_raw_all['Space'].isin(list_space)]



##### Step2: Create a grouping level for each category
list_category = df_raw_all['Category'].unique().tolist()
print(list_category)

### Saperate into different level group
# For category using space
list_space_level = [ 
                    # Soft drinks
                      'SOFT DRINKS'
                    , 'SOFT DRINKS, BOTTLED'
                    , 'SOFT DRINKS, CANNED'
                    , 'SOFT DRINKS, MIXERS'
                    , 'SOFT DRINKS, SINGLE'
                    # Beer
                    , 'BEER - FULL STRENGTH'
                    # Red Wines
                    , 'WINES RED - LOCAL'
                    , 'WINES RED, BOTTLED'
                    # White Wine
                    , 'WINES ROSE - LOCAL'
                    , 'WINES WHITE - LOCAL']

# For category using category only
list_category_level = ['BISCUITS'
                       ,'CHOCOLATES'
                       ,'SNACKS AND SAVOURIES']

# For category using new category names
list_confectionery = [
                      'CONFECTIONERY'
                     , 'CONFECTIONERY, CHEWI'
                     ]

# Health food
list_heal_food =    [
                      'HEALTH FOODS'
                    , 'HEALTH FOODS ORGANIC'
                    ]

### Create new column for grouping
# Setup conditions
conditions = [
                df_raw_all['Category'].isin(list_space_level)
                ,df_raw_all['Category'].isin(list_category_level)
                ,df_raw_all['Category'].isin(list_confectionery)
                ,df_raw_all['Category'].isin(list_heal_food)
                ]
# Setup values
choices = [
            df_raw_all['Space']
            ,df_raw_all['Category']
            ,'CONFECTIONERY'
            ,'HEALTH FOODS'
            ]

# Fill in Space value for list_space_level
# np.select(conditions, choices, default=0)
df_raw_all['Calculation Group'] = np.select(conditions, choices, default = 0)



##### Step3: Remove out of scope category
# Remove health food category, as the prices don't change a lot
fil_health_food = df_raw_all['Calculation Group'] == 'HEALTH FOODS'
df_raw_all = df_raw_all[~fil_health_food]



##### Step4: Check Grouping result
df_check_group = df_raw_all[['Department','Space','Category','Calculation Group']].drop_duplicates()



##### Step4: Change week numbers from float to int
df_raw_all['Week_number'] = df_raw_all['Week_number'].astype(int)

# Filter data within Q1 Week number < 13
fil_q1 = df_raw_all['Week_number']<13

df_raw_all = df_raw_all[fil_q1]

# Filter out store = na
df_raw_all = df_raw_all[~df_raw_all['Store'].isna()]

######### End of Block2: Process Dataframe- Combine category ##########





######### Block3: Exclude outliers by indivisual's threshold ##########

##### Step1: Read reference table
### Create path files
fld_root_analysis = r"os.path.join(DATA_DIR, '')"
file_ref_threshold = fld_root_analysis + "Reference_Threshold.xlsx"
print(file_ref_threshold)

### Read reference table
df_ref_threshold = pd.read_excel(file_ref_threshold
                                 ,sheet_name="Threshold>1_Week<=6_WkW2")



##### Step2: Create loops to process threshold by different level
dfs_raw_apply_thresolds = []
for store in df_raw_all['Store'].unique():
    # df for sales dataset
    df_store = df_raw_all[df_raw_all['Store'] == store]
    
    # df for reference table
    df_ref_store = df_ref_threshold[df_ref_threshold['Store'] == store]
    print(store)
    for category in df_store['Calculation Group'].unique():
        # df for sales dataset
        df_store_category = df_store[df_store['Calculation Group'] == category]
        # df for reference table
        df_ref_store_category = df_ref_store[df_ref_store['Calculation Group'] == category]
        print(category)
        for qty, wk in zip(df_ref_store_category['Threshold'], df_ref_store_category['Weeks above threshold']):
            print(qty, wk)
            # Add column to check whether QTY >= threshold
            df_store_category['Greater than threshold'] = df_store_category['QTY'] >= qty

            # Sum by store, product
            df_ref_abv_thrlds = df_store_category.groupby(['Store','Product'])['Greater than threshold'].sum().reset_index()
            df_ref_abv_thrlds = df_ref_abv_thrlds.rename(columns={'Greater than threshold':'Weeks above threshold'})

            # Apply threshold for weeks
            fil_wk_abv_thrlds = df_ref_abv_thrlds['Weeks above threshold'] >= wk

            df_ref_abv_thrlds = df_ref_abv_thrlds[fil_wk_abv_thrlds]

            # Make the group by df as a reference table
            df_store_category_abv_thrlds = pd.merge(   df_store_category
                                                      ,df_ref_abv_thrlds
                                                      ,how='inner'
                                                      ,on=['Store', 'Product'])

            # Append all processed dfs together
            dfs_raw_apply_thresolds.append(df_store_category_abv_thrlds)

# Concat all dfs together as df_raw_all
df_raw_all = pd.concat(dfs_raw_apply_thresolds)

######## End of Block3: Exclude outliers by indivisual's threshold ##########




######## Block4: Filter by products without price change ##########

##### Step1: Define function to check if there is only one unique price in the group
def has_price_change(x):
    return x.nunique() > 1


##### Step2: Apply function
df_raw_all = df_raw_all.groupby(['Store', 'Product']).filter(lambda x: has_price_change(x['AVG_Price']))

### Check filtered result
df_raw_all['Store'].unique()

######## End of Block4: Filter by products without price change ##########





######## Block5: Create loops to calculate correlation between products in price and quantity ##########

##### Step1: Create a folder of analysis
fld_root_analysis = r"os.path.join(DATA_DIR, '')"

# Initialize an empty list to store the results
results = []

##### Step2-1: Outer loop, subset df by stores 
for store in df_raw_all['Store'].unique():
    
    # df for sales dataset
    df_store = df_raw_all[df_raw_all['Store'] == store]
    print(store)
 
    ##### Step2-2: Inner loop, subset store df by category
    for category in df_store['Calculation Group'].unique():
        # df for sales dataset
        df_store_category = df_store[df_store['Calculation Group'] == category]
        print(category)

        # Get unique products
        products = df_store_category["Product"].unique()
                
        ##### Step2-3: Inner category loop, subset category df by product
        ### Loop for getting all combinations of product
        # i = index, product_1 will be product index = 0
        for i, product_1 in enumerate(products):
            # product_2 will start from 0 - n. Meaning the inner loop will fetch the next product
            # Check that the indices i and j are not equal, ensuring that product_2 is never the same as product_1.
            for j, product_2 in enumerate(products):
                if i != j:
                    print(f"product_1: {product_1}, product_2: {product_2}")
                    df_p1 = df_raw_all[df_raw_all["Product"] == product_1].sort_values(by="Week_number")
                    df_p2 = df_raw_all[df_raw_all["Product"] == product_2].sort_values(by="Week_number")
        
                    # Merge the data frames on the Week_number column
                    df_merged = pd.merge(  df_p1
                                         , df_p2
                                         , on="Week_number"
                                         , suffixes=(f'_{product_1}', f'_{product_2}'))
            
                    # Calculate the correlation coefficient
                    correlation_p_p = df_merged[f"AVG_Price_{product_1}"].corr(df_merged[f"AVG_Price_{product_2}"])
                    correlation_q_q = df_merged[f"QTY_{product_1}"].corr(df_merged[f"QTY_{product_2}"])
                    total_correlation = correlation_p_p + correlation_q_q
                    
                    # Add the result to the list
                    results.append({
                                    "Store": store,
                                    'Calculation Group': category,
                                    "Product_1": product_1,
                                    "Product_2": product_2,
                                    "Correlation_Coefficient_P": correlation_p_p,
                                    "Correlation_Coefficient_Q": correlation_q_q,
                                    "Total_Correlation": total_correlation
                                    })
            
# Convert the results to a DataFrame
results_df = pd.DataFrame(results)

# Output
file_corr_result = fld_root_analysis + "Correlation_By_StoreTypes.xlsx"
print(file_corr_result)
results_df.to_excel(file_corr_result, header=True, index=False)

######## End of Block5: Create loops to calculate correlation between products in price and quantity ##########




