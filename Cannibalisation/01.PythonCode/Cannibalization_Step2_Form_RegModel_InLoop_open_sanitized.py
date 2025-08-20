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
            """
            Return pairs of (product_id, competitor_id) that likely interact.
            Public-safe default: returns empty list (no cannibalisation).
            """
            return []

        @staticmethod
        def build_demand_model(train_df, features, target):
    return _strategy.build_demand_model(train_df, features, target)

import pandas as pd
import datetime as dt
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from scipy.stats import linregress
try:
    from sklearn import linear_model
except Exception:
    linear_model=None.linear_model import LinearRegression
try:
    from sklearn import linear_model
except Exception:
    linear_model=None.metrics import mean_squared_error
try:
    from statsmodels import api as sm
except Exception:
    sm=None.stats.outliers_influence import variance_inflation_factor
try:
    import statsmodels
except Exception:
    statsmodels=None.api as sm

# Define a function to replace invalid characters
import re
def sanitize_filename(name):
    return re.sub(r'[\/:*?"<>|]', '_', name)
          
# Define a function to round up a value to the next integer
def round_up(x):
    return math.ceil(x)



########## Block 1: Read raw data ##########

##### Step 1: Read dataset
# File path
fld_data = r"os.path.join(DATA_DIR, '')"
file_daily_sales = fld_data + "DailyQTY_SelectedCategories_24Q1_FillNA.csv"
print(file_daily_sales)

# Read file
df_raw_complete = pd.read_csv(file_daily_sales)



##### Step 2: Check date range
max_date = pd.to_datetime(df_raw_complete['Date'],format='%Y-%m-%d').max()
min_date = pd.to_datetime(df_raw_complete['Date'],format='%Y-%m-%d').min()

### Check store list
list_str = df_raw_complete['Store'].unique().tolist()

############# End of Block 1: Read raw data ##########





############# Block 2: Calculate features for simple regression model #############

##### Step 1: Copy dataset for simple regression model
df_pnq_smp_reg = df_raw_complete.copy()



##### Step 2: Count the unique prices for each product
df_pnq_smp_reg['Unique_Prices'] = df_pnq_smp_reg.groupby(['Store'
                                                          ,'Product'])['Price'].transform('nunique')



##### Step 3: Count the unique days for each price
df_pnq_smp_reg['Unique_Days'] = df_pnq_smp_reg.groupby(['Store'
                                                        ,'Product'
                                                        ,'Price'])['Date'].transform('nunique')



##### Step 4: Calculate the total 'QTY' for each price
df_pnq_smp_reg['Total_QTY'] = df_pnq_smp_reg.groupby(['Store'
                                                      ,'Product'
                                                      ,'Price'])['QTY'].transform('sum')



##### Step 5: Calculate the weighted average of 'QTY' based on 'Unique_Days'
df_pnq_smp_reg['Avg_QTY'] = df_pnq_smp_reg['Total_QTY'] / df_pnq_smp_reg['Unique_Days']



##### Step 6: Filter columns
df_pnq_smp_reg = df_pnq_smp_reg[[
                                  'Store'
                                 ,'Calculation Group'
                                 ,'Product'
                                 ,'Price'
                                 ,'Avg_QTY'
                                 ,'Unique_Prices'
                                 ]]



##### Step 7: Drop duplicates
df_pnq_smp_reg.drop_duplicates(inplace=True)



##### Step 8: Sort the DataFrame by 'Product' and 'Price'
df_pnq_smp_reg.sort_values(['Store'
                            , 'Product'
                            , 'Price'], ascending=False, inplace=True)



##### Step 9: Calculate the "Inverted_QTY" column
df_pnq_smp_reg['Inverted_QTY'] =  df_pnq_smp_reg.groupby(['Store'
                                                          ,'Product'
                                                          ])['Avg_QTY'].cumsum()



##### Step 10: Sort the DataFrame by 'Product' and 'Price'
df_pnq_smp_reg.sort_values(['Store', 'Product', 'Price'], inplace=True)

############# End of Block 2: Calculate features for simple regression model #############




############# End of Block 3: Train regression models #############

##### Step1: Initialize lists to store measurements
store_smpl_reg = []
category_smpl_reg = []
product_smpl_reg = []
coefficients_smpl_reg = []
intercepts_smpl_reg = []
independent_var_smpl_reg = []
p_values_smpl_reg = []
r_squared_smpl_reg = []
linearity_smpl_reg = []



##### Step2: Loop for training simple regression

##### Step2-1: Loop for subsetting dfs

### Outer loop: filter by store
for store in df_pnq_smp_reg['Store'].unique():
    # df for sales dataset
    df_store = df_pnq_smp_reg[df_pnq_smp_reg['Store'] == store]
    print(store)
    
    ### Inner loop1: filter by Calculation Group under store
    for category in df_store['Calculation Group'].unique():
        # df for sales dataset
        df_store_category = df_store[df_store['Calculation Group'] == category]
        print(category)

        ### Inner loop2: filter by Product under Calculation Group
        for product in df_store_category['Product'].unique():
            df_str_cat_prd = df_store_category[df_store_category["Product"] == product]
            print(product)
            
            ### Check if there are enough data points for regression
            if len(df_str_cat_prd) < 2:
                print(f"Not enough data points for {store} - {product}. Skipping.")
                
                ### Store measurements and linearity in the respective lists
                store_smpl_reg.append(store)
                product_smpl_reg.append(product)
                linearity_smpl_reg.append('Less than 2 data points')
                
                continue
            
            ##### Step2-2: Under product, train model
            
            ### Prepare the independent and dependent variables
            X_smp_reg = df_str_cat_prd['Price']
            y_smp_reg = df_str_cat_prd['Inverted_QTY']
        
            ### Add a constant to the independent variables for the intercept
            X_smp_reg = sm.add_constant(X_smp_reg)
        
            ### Fit the model using /* redacted: proprietary modelling detail */
            model = sm./* redacted: proprietary modelling detail */(y_smp_reg,X_smp_reg).fit()
            
            
            ##### Step2-3: Extract model information

            ### Replace inf, -inf with NaN in model parameters
            model_params = model.params.replace([np.inf, -np.inf], np.nan)
            model_pvalues = model.pvalues.replace([np.inf, -np.inf], np.nan)
            model_rsquared = np.nan if model.rsquared == np.inf else model.rsquared


            ##### Step2-4: Identify linearity

            ### Check if the relationship is linear based on the p-value
            alpha = 0.05  # Set a significance level

            # extract p-values
            for i, col_nm in enumerate(X_smp_reg.columns):
                
                p_value_smp_reg = model_pvalues[i]
                
                if p_value_smp_reg < alpha:
                    relationship_smp_reg = "Linear"
                else:
                    relationship_smp_reg = "Not Linear"


            ##### Step2-5: Store measurements and linearity in the respective lists
            store_smpl_reg.append(store)
            category_smpl_reg.append(category)
            product_smpl_reg.append(product)
            independent_var_smpl_reg.append(col_nm)
            coefficients_smpl_reg.append(model_params[col_nm])
            intercepts_smpl_reg.append(model_params['const'])
            p_values_smpl_reg.append(p_value_smp_reg)
            r_squared_smpl_reg.append(model_rsquared)
            linearity_smpl_reg.append(relationship_smp_reg)


##### Step2-6: Create a DataFrame to store results for each variable
df_result_smp_reg = pd.DataFrame({
                                    'Store': store_smpl_reg,
                                    'Calculation Group':category_smpl_reg,
                                    'Product': product_smpl_reg,
                                    'Variable': independent_var_smpl_reg,
                                    'Coefficient': coefficients_smpl_reg,
                                    'Intercept': intercepts_smpl_reg,
                                    'P-Value': p_values_smpl_reg,
                                    'R-Squared': r_squared_smpl_reg,
                                    'Linearity': linearity_smpl_reg
                                })


##### Step2-7: Group models based on R-squared values

### Set up bins and labels for cut function
bins = [0, 0.3, 0.8, 1.0]
labels = ['Low', 'Medium', 'High']

### Apply cut function
df_result_smp_reg['R_squared_Group'] = pd.cut(  # The measurement you want to cut
                                                df_result_smp_reg['R-Squared']
                                                # apply bins(groups)
                                              , bins=bins
                                                # apply labels
                                              , labels=labels
                                              , include_lowest=True)


##### Step2-8: Handel na

# Add a category to the list of category varialbe
df_result_smp_reg['R_squared_Group'] = df_result_smp_reg['R_squared_Group'].cat.add_categories(['No_R-Square'])

# Fill na
df_result_smp_reg['R_squared_Group'] = df_result_smp_reg['R_squared_Group'].fillna('No_R-Square')

############# End of Block 3: Train regression models #############





########## Block 4: Apply & filter cannibalisation thresholds on price and quantity ##########

##### Step0: Read reference table
### Previously correlation output as all stores
fld_result = r"os.path.join(DATA_DIR, '')"

### Define file path
file_ref = fld_result + "Correlation_AllStoreTypes_+11_P_0.3_Q_0.2.csv"

### Read file
df_ref_cnnbl = pd.read_csv(file_ref)



##### Step1: Apply different threshold to different store types

### Conditions for Store Type
conditions = [df_ref_cnnbl['Store'] == 'Country Store'
              ,df_ref_cnnbl['Store'] == 'Metro Store'
              ,df_ref_cnnbl['Store'] == 'Metro Discount']

### Choices for threshold_P
choices_p = [-0.3
             ,-0.3
             ,-0.35]

### Choices for threshold_Q
choices_q = [-0.3
             ,-0.25
             ,-0.35]

### Apply conditions & choices
# For Price
df_ref_cnnbl['Threshold_P'] = np.select(conditions
                                        ,choices_p
                                        ,default=-1
                                       )

# For QTY
df_ref_cnnbl['Threshold_Q'] = np.select(conditions
                                        ,choices_q
                                        ,default=-1
                                       )

# Check result
df_check_ref_thrld = df_ref_cnnbl[['StoreType',
                                   'Threshold_P',
                                   'Threshold_Q']].drop_duplicates()



##### Step2: Apply threshold
df_ref_cnnbl.dtypes

# Filter for P
fil_threshold_P = df_ref_cnnbl['Correlation_Coefficient_P'] < df_ref_cnnbl['Threshold_P']

# Filter for Q
fil_threshold_Q = df_ref_cnnbl['Correlation_Coefficient_Q_Rank'] < df_ref_cnnbl['Threshold_Q']

# Apply filters
df_ref_cnnbl_filtered = df_ref_cnnbl[ fil_threshold_P & fil_threshold_Q ]



##### Step3: Keep necessary columns

### Create column list
list_cnnbl_sort_order = [
                        'Store'
                        ,'Calculation Group'
                        ,'Product_1'
                        ,'Correlation_Coefficient_Q_Rank'
                        ]

### Apply and sort values
df_ref_cnnbl_filtered = df_ref_cnnbl_filtered.sort_values(by=list_cnnbl_sort_order
                                                         ,ascending=True)

### Columns to keep
list_col_ref_cnnbl = ['Store'
                      ,'Product_1'
                      ,'Product_2'
                      ]

# Apply filter
df_ref_cnnbl_filtered = df_ref_cnnbl_filtered[list_col_ref_cnnbl]

########## End of Block 4: Apply & filter cannibalisation thresholds on price and quantity ##########





########## Block 5: Adding prices of cannibalic products ##########
########## Block 6: Loop: Pivot table & Train multi-variance regression model  ########## 
########## Block 7: Aggregate qty by ##########
########## Block 8: Create inverted QTY ##########
########## Block 9: Model Training ##########

###### Need to use loop to process the dataset otherwise will be out of memory

##### Step1: Initiate lists to store all dfs

### Initialize list for dfs
dfs_raw_all_cnnbl = []

### Initialize an empty list to store the results
results = []

### Initialize lists to store measurements from model
store_list = []
category_list = []
product_list = []
coefficients = []
intercepts = []
p_values_list = []
r_squared_values = []
linearity = []
/* redacted: proprietary modelling detail */_values = []
independent_var_names = []



##### Step2: Create loops to add prices of cannibalic products into 1 df

for store in df_raw_complete['Store'].unique():
    # df for sales dataset
    df_raw_complete_str = df_raw_complete[df_raw_complete['Store'] == store]
    df_ref_cnnbl_filtered_str = df_ref_cnnbl_filtered[df_ref_cnnbl_filtered['Store'] == store]
    print(store)
    
    
    
    ##### Step3: Daily sales dataset merges with ref of cnnbl to get product 2
    df_raw_all_cnnbl_str = pd.merge(
                                    # df_raw_all # only contains txn with QTY
                                    df_raw_complete_str
                                    ### join df with product1 & 2 only
                                   ,df_ref_cnnbl_filtered_str
                                   ,how="inner"
                                   ,left_on = ['Store','Product']
                                   ,right_on = ['Store','Product_1']
                                   )
    
    ### Drop the duplicate column
    df_raw_all_cnnbl_str = df_raw_all_cnnbl_str.drop(columns=['Product_1'])
    
    
    
    ##### Step4: Use result of Step1 to get price of Product2 and add prefix
    ### Reference talbe for date and price only
    df_raw_all_ref_str = df_raw_complete_str[[ 'Store'                             
                                             , 'Product'
                                             , 'Calculation Group'
                                             , 'Date'
                                             ,'Price']].drop_duplicates()



    ##### Step5: Merge with reference table to get the price of product 2
    df_raw_all_cnnbl_str = pd.merge(
                                    df_raw_all_cnnbl_str
                                   ,df_raw_all_ref_str
                                   ,how="left"
                                   ,left_on = ['Store','Calculation Group','Product_2','Date']
                                   ,right_on = ['Store','Calculation Group','Product','Date']
                                   ,suffixes=('','_Cannibalized')
                                   )
    
    ### Drop columns
    df_raw_all_cnnbl_str = df_raw_all_cnnbl_str.drop(columns=['Product_Cannibalized'])
    
    
    
    ##### Step6: Add price difference
    df_raw_all_cnnbl_str['PriceDifference'] = df_raw_all_cnnbl_str['Price'] - df_raw_all_cnnbl_str['Price_Cannibalized']
    
    
    
    ########## Block 6: Loop: Pivot table & Train multi-variance regression model 

    ##### Step1: Subset by Calculation Group within store
    for category in df_raw_all_cnnbl_str['Calculation Group'].unique():
        
        ### df for sales dataset
        df_store_category = df_raw_all_cnnbl_str[df_raw_all_cnnbl_str['Calculation Group'] == category]
        print(category)

        ### Get unique products df
        for product in df_store_category['Product'].unique():
            df_str_cat_prd = df_store_category[df_store_category["Product"] == product]
            print(product)
            
            ### df product
                                                 #Index: Columns that you want to remain as they are
            df_pivot = df_str_cat_prd.pivot_table(index=['Store'
                                                       ,'Department'
                                                       ,'Space'
                                                       ,'Category'
                                                       ,'Calculation Group'
                                                       ,'Product'
                                                       ,'Date'
                                                       ,'Price'
                                                       ,'QTY'
                                                       ,'Original_Ind'
                                                       ]
                                                #Column values that will become the new columns
                                                ,columns='Product_2'
                                                #The values you want to fill in the new columns
                                                # Use absolute price once
                                                # ,values='Price_Cannibalized'
                                                ,values='PriceDifference'
                                                )
            ### Reset index
            df_pivot.reset_index(inplace=True)
            
            ### Create list of cnnbl pirce
            # List of all columns
            list_col_all = df_pivot.columns.tolist()
            
            ### Remove list from 
            list_col_cnbl_prd = list_col_all[10:]
            
            ### Adding all price diff in 1 column
            df_pivot['PriceDiff_All'] = df_pivot[list_col_cnbl_prd].sum(axis=1)
            
            
            
            
            
            ########## Block 7: Aggregate qty by ##########
            
            ##### Create list for different aggregation
            ### List of all columns
            list_pivot_all = df_pivot.columns.values.tolist()
            # print(list_pivot_all)
            
            ### List without price
            list_wo_price = ['Store'
                             , 'Department'
                             , 'Space'
                             , 'Category'
                             , 'Calculation Group'
                             , 'Product']
            
            # Count the unique prices for each product
            df_pivot['Unique_Prices'] = df_pivot.groupby(list_wo_price)['Price'].transform('nunique')
            
            ### List without date
            list_wo_date = [item for item in list_pivot_all if item not in ['Date', 'QTY', 'Original_Ind']]
            # print(list_wo_date)
            
            # Count the unique days for each price
            df_pivot['Unique_Days'] = df_pivot.groupby(list_wo_date)['Date'].transform('nunique')
            
            # Calculate the total 'QTY' for each price
            df_pivot['Total_QTY'] = df_pivot.groupby(list_wo_date)['QTY'].transform('sum')
            
            # Calculate the weighted average of 'QTY' based on 'Unique_Days'
            df_pivot['Avg_QTY'] = df_pivot['Total_QTY'] / df_pivot['Unique_Days']
            
            # Remove duplicate rows
            df_pnq = df_pivot.copy()
            
            # Create list of columns for ABT
            list_pivot_agg = df_pnq.columns.values.tolist()
            # print(f"List of pivot agg: {list_pivot_agg}")
            
            # Keep related columns
            list_abt = [item for item in list_pivot_agg if item not in ['Date', 'QTY', 'Original_Ind'] ]
            # print(f"List of abt: {list_abt}")
            
            # Print or further analyze the result
            df_pnq = df_pnq[list_abt]
            
            df_pnq.drop_duplicates(inplace=True)
            
            ########## End of Block 7: Aggregate qty by ##########





            ########## Block 8: Create inverted QTY ##########
            # Create sort values list
            list_abt_price = list_abt[6:-4]
            # print(f"List of prices: {list_abt_price} ")
            
            #  Sort the DataFrame by 'Product' and 'Price' and cannabilized prices
            df_pnq.sort_values(list_abt_price, ascending=False, inplace=True)

            # Calculate the "Inverted_QTY" column
            df_pnq['Inverted_QTY'] =  df_pnq.groupby(['Store', 'Product'])['Avg_QTY'].cumsum()
            
            # Drop nan to train the model
            # df_pnq = df_pnq.dropna() # Renote
            
            ########## End of Block 8: Create inverted QTY ##########
            
            
            
            
            
            ########## Block 9: Model Training ##########

            ##### Step1: Function to calculate /* redacted: proprietary modelling detail */ to avoid co-linearity
            def calculate_/* redacted: proprietary modelling detail */(X):
                # Create an empty data frame
                /* redacted: proprietary modelling detail */_data = pd.DataFrame()
                # Get columns and save as a column, .columns returns an array of index
                /* redacted: proprietary modelling detail */_data['Variables'] = X.columns
                ### Create a new column of /* redacted: proprietary modelling detail */ by its function
                # DF.shape -> returns (# of rows, # of c/* redacted: proprietary modelling detail */)
                # /* redacted: proprietary modelling detail */(array of each row, index of columns)
                /* redacted: proprietary modelling detail */_data['/* redacted: proprietary modelling detail */'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                return /* redacted: proprietary modelling detail */_data
            
            
            
            ##### Step2: Function loop to remove variable with /* redacted: proprietary modelling detail */ > 5
            def remove_high_/* redacted: proprietary modelling detail */(X, threshold=5):
                ### By the author of /* redacted: proprietary modelling detail */ calculation
                ### variance_inflation_factor under the assumption that a regression that already includes a constant.
                # Add constant once before the loop
                X_with_const = sm.add_constant(X)
                
                ### /* redacted: proprietary modelling detail */ test
                while True:
                    # use set to avoid Order Sensitive & extra / missing c/* redacted: proprietary modelling detail */
                    if set(X_with_const.columns) == {'const', 'Price'}:
                        print("Only price and constant left, no /* redacted: proprietary modelling detail */ calculation possible.")
                        break
                    
                    # if the col check is not true, then calculate /* redacted: proprietary modelling detail */
                    /* redacted: proprietary modelling detail */_data = calculate_/* redacted: proprietary modelling detail */(X_with_const)
                    
                    # Identify high /* redacted: proprietary modelling detail */ variables, excluding 'Price'
                    high_/* redacted: proprietary modelling detail */_vars = /* redacted: proprietary modelling detail */_data[/* redacted: proprietary modelling detail */_data['/* redacted: proprietary modelling detail */'] > threshold]['Variables'].tolist()
                    
                    # ensures that if there are no high /* redacted: proprietary modelling detail */ variables left to process
                    if high_/* redacted: proprietary modelling detail */_vars:
                        # Exclude 'const' & 'Price' from removal
                        high_/* redacted: proprietary modelling detail */_vars = [var for var in high_/* redacted: proprietary modelling detail */_vars if var not in ['const','Price']]
                        
                        if len(high_/* redacted: proprietary modelling detail */_vars) > 0:
                            print(f"High /* redacted: proprietary modelling detail */ Variables: {high_/* redacted: proprietary modelling detail */_vars}")
                            X_with_const = X_with_const.drop(columns=high_/* redacted: proprietary modelling detail */_vars)
                        else:
                            # No more variables left to remove
                            print("No more variables with high /* redacted: proprietary modelling detail */ left to remove.")
                            break
                    else:
                        # All /* redacted: proprietary modelling detail */ values are below the threshold, exit the loop
                        break
                        
                return X_with_const
            
            
            
            ##### Step3: Function looop to remove variable with P-value > 0.05            
            def remove_high_p_values(X_with_const, y, alpha=0.05):
                
                ### Loop for p-value test
                while True:
                    model = sm./* redacted: proprietary modelling detail */(y, X_with_const).fit()
                    p_values = model.pvalues
                    
                    # Check if any p-values are greater than the alpha level
                    if p_values.max() > alpha:
                        high_p_values_vars = p_values[p_values > alpha].index.tolist()
                        
                        # ensures that if there are no high p-value variables left to process
                        if high_p_values_vars:
                            print(f"High P-value Variables: {high_p_values_vars}")
                        
                            # Ensure we don't remove 'const' and 'Price'
                            high_p_values_vars = [var for var in high_p_values_vars if var not in ['const', 'Price']]
                            
                            # If only 'const' and 'Price' are left, break to avoid infinite loop
                            if len(high_p_values_vars) == 0:
                                print("No more removable variables with high p-values left.")
                                break
                            
                            # Remove the variables with high p-values
                            X_with_const = X_with_const.drop(columns=high_p_values_vars)
                    else:
                        break
                
                return X_with_const, model

            

            ##### Step4: Prepare independent and dependent variables
            X = df_pnq[list_abt_price]
            y = df_pnq['Inverted_QTY']
            
            
            
            ##### Step5: Check if there are enough data points
            if len(df_pnq) < 2:
                print(f"Not enough data points for {store} - {product}. Skipping.")
            else:
                # Handle /* redacted: proprietary modelling detail */
                X_with_const  = remove_high_/* redacted: proprietary modelling detail */(X)
            
                # Handle p-values
                X_with_const , model = remove_high_p_values(X_with_const , y)


            
                ##### Step6: Iterate column names
                for idx, var_name in enumerate(X_with_const.columns):
                    p_val = model.pvalues[idx]
                    linearity_check = "Linear" if p_val < alpha else "Not Linear"
                    
                    # Calculate /* redacted: proprietary modelling detail */ for each independent variable
                    if var_name != 'const':  # Exclude intercept from /* redacted: proprietary modelling detail */ calculation
                        /* redacted: proprietary modelling detail */ = variance_inflation_factor(X_with_const.values, X_with_const.columns.get_loc(var_name))
                    else:
                        /* redacted: proprietary modelling detail */ = np.nan  # Skip /* redacted: proprietary modelling detail */ calculation for intercept


            
                    ##### Step7: Store the results
                    store_list.append(store)
                    category_list.append(category)
                    product_list.append(product)
                    independent_var_names.append(var_name)
                    coefficients.append(model.params[var_name])
                    intercepts.append(model.params['const'])
                    p_values_list.append(p_val)
                    r_squared_values.append(model.rsquared)
                    linearity.append(linearity_check)
                    /* redacted: proprietary modelling detail */_values.append(/* redacted: proprietary modelling detail */)



##### Step8: Create a DataFrame to store results for each variable
df_result_/* redacted: proprietary modelling detail */ = pd.DataFrame({
                                'Store': store_list,
                                'Calculation Group':category_list,
                                'Product': product_list,
                                'Variable': independent_var_names,
                                'Coefficient': coefficients,
                                'Intercept': intercepts,
                                'P-Value': p_values_list,
                                'R-Squared': r_squared_values,
                                'Linearity': linearity,
                                '/* redacted: proprietary modelling detail */': /* redacted: proprietary modelling detail */_values
                            })



##### Step9: Group models based on R-squared values

### Set up parameters for cut function
bins = [0, 0.3, 0.8, 1.0]
labels = ['Low', 'Medium', 'High']

### Apply cut function
df_result_/* redacted: proprietary modelling detail */['R_squared_Group'] = pd.cut(  # The measurement you want to cut
                                                df_result_/* redacted: proprietary modelling detail */['R-Squared']
                                                # apply bins(groups)
                                              , bins=bins
                                                # apply labels
                                              , labels=labels
                                              , include_lowest=True)



##### Step10: Handel nan

### Add a category to the list of category varialbe
df_result_/* redacted: proprietary modelling detail */['R_squared_Group'] = df_result_/* redacted: proprietary modelling detail */['R_squared_Group'].cat.add_categories(['No_R-Square'])

# Fill na
df_result_/* redacted: proprietary modelling detail */['R_squared_Group'] = df_result_/* redacted: proprietary modelling detail */['R_squared_Group'].fillna('No_R-Square')

########## End of Block 5: Adding prices of cannibalic products ##########
########## End of Block 6: Loop: Pivot table & Train multi-variance regression model  ########## 
########## End of Block 7: Aggregate qty by ##########
########## End of Block 8: Create inverted QTY ##########
########## End of Block 9: Model Training ##########





########## Block 10: Join models into 1 table to compare ##########

##### Step1: Create list of col on product level
list_col_join = ['Store'
                ,'Calculation Group'
                ,'Product'
                ,'Variable']



##### Step2: Join together to compare
df_model_compare = pd.merge(df_result_smp_reg
                            ,df_result_/* redacted: proprietary modelling detail */
                            ,how='left'
                            ,on=list_col_join
                            ,suffixes=('_SimpleRegression','_/* redacted: proprietary modelling detail */'))



##### Step3: Add indicator and lift

### Add Cannibalization_IND
df_model_compare['Cannibalization_IND'] = df_model_compare['R-Squared_/* redacted: proprietary modelling detail */'].apply(lambda x: 0 if pd.isna(x) else 1)

### Calculate lift
df_model_compare['Lift_by_Cannibalization'] = df_model_compare['R-Squared_/* redacted: proprietary modelling detail */'] - df_model_compare['R-Squared_SimpleRegression']

### Create uplift ind
df_model_compare['UpLift_IND'] = df_model_compare['Lift_by_Cannibalization'].apply(lambda x: 1 if x>0 else 0)



##### Step4: Add product features
# Copy dataset for simple regression model
df_prd_features = df_raw_complete.copy()

# Create indicator for day with sales
df_prd_features['Day_with_sales_IND'] = df_prd_features['QTY'].apply(lambda x: 1 if x>0 else 0)

# Sum the indicator
df_prd_features['Days_with_Sales'] = df_prd_features.groupby(['Store'
                                                              ,'Product'])['Day_with_sales_IND'].transform('sum')

# Count the unique prices for each product
df_prd_features['Unique_Prices'] = df_prd_features.groupby(['Store'
                                                          ,'Product'])['Price'].transform('nunique')

# Count the unique days for each price
df_prd_features['Unique_Days'] = df_prd_features.groupby(['Store'
                                                        ,'Product'])['Date'].transform('nunique')

# Calculate the total 'QTY' for each price
df_prd_features['Total_QTY'] = df_prd_features.groupby(['Store'
                                                      ,'Product'])['QTY'].transform('sum')

# Calculate the weighted average of 'QTY' based on 'Unique_Days'
df_prd_features['Avg_QTY'] = df_prd_features['Total_QTY'] / df_prd_features['Unique_Days']

# Drop duplicates
df_prd_features = df_prd_features[['Store'
                                    ,'Calculation Group'
                                    ,'Product'
                                    ,'Unique_Prices'
                                    ,'Days_with_Sales'
                                    ,'Total_QTY'
                                    ,'Avg_QTY']].drop_duplicates()



##### Step5: Join product features
### Merge 2 tables
df_model_compare = pd.merge(df_prd_features
                            ,df_model_compare
                            ,how='left'
                            ,on=['Store','Calculation Group','Product']
                            )

# Adding row numbers
df_model_compare['Row_ID'] =  range(1, len(df_model_compare) + 1)

# Join product features
df_result_smp_reg_add_prd = pd.merge(df_result_smp_reg
                                    ,df_prd_features
                                    ,how='left'
                                    ,on=['Store','Calculation Group','Product']
                                    )

# Join product features
df_result_/* redacted: proprietary modelling detail */_add_prd = pd.merge(df_result_/* redacted: proprietary modelling detail */
                                ,df_prd_features
                                ,how='left'
                                ,on=['Store','Calculation Group','Product']
                                )


########## End of Block 10: Join models into 1 table to compare ##########





########## Block 11: Save all result to excel
### Save result to Excel
# file_model_result = fld_analysis + "ModelComparison_+11.xlsx"
# file_model_result = fld_analysis + "ModelComparison_+11_Threshold_0_Check_/* redacted: proprietary modelling detail */_P_Const.xlsx"
file_model_result = fld_analysis + "ModelComparison_+11_Threshold_0.25_Check_/* redacted: proprietary modelling detail */_P_Const.xlsx"
print(file_model_result)

# Output to excel file
with pd.ExcelWriter(file_model_result, engine='xlsxwriter') as writer:
    # Write each DataFrame to a different sheet
    df_model_compare.to_excel(writer, sheet_name='Model_Comparison', index=False, header=True)
    df_result_smp_reg_add_prd.to_excel(writer, sheet_name='SimpleRegression', index=False, header=True)
    df_result_/* redacted: proprietary modelling detail */_add_prd.to_excel(writer, sheet_name='/* redacted: proprietary modelling detail */', index=False, header=True)

########## End of Block 11: Save all result to excel







######## Check Linearity & find variables importance
df_model_compare.dtypes
df_result_smp_reg_add_prd.dtypes
df_result_/* redacted: proprietary modelling detail */_add_prd.dtypes

try:
    from sklearn import linear_model
except Exception:
    linear_model=None.tree import DecisionTreeClassifier, plot_tree
try:
    from sklearn import linear_model
except Exception:
    linear_model=None.model_selection import train_test_split
try:
    from sklearn import linear_model
except Exception:
    linear_model=None import metrics
try:
    from sklearn import linear_model
except Exception:
    linear_model=None.preprocessing import OneHotEncoder
try:
    from sklearn import linear_model
except Exception:
    linear_model=None import tree
try:
    from sklearn import linear_model
except Exception:
    linear_model=None.tree import export_graphviz
import graphviz

##### Check if there's any NaN in dfs

# Loop to find columns with nan
list_col_contains_na = []

# for col in X_encoded.columns.unique():
for col in df_model_compare.columns.unique():
    # df[col].isna() returns a series of boolean value. Add .any() to return T/F
    if df_model_compare[col].isna().any():
    # if X_encoded[col].isna().any():
        list_col_contains_na.append(col)
        print(f"{col} contains NaN.")

# Check rows where 'R_squared_Group_SimpleRegression' is NaN
df_check_na = df_model_compare[df_model_compare['R_squared_Group_SimpleRegression'].isna()]

##### Check if there's any NaN in dfs


##### Prepare features & Train model

# Filter explanary variables
X = df_model_compare[['Store'
                      , 'Calculation Group'
                      , 'Cannibalization_IND'
                      , 'Unique_Prices'
                      , 'Total_QTY'
                      , 'Avg_QTY'
                      , 'Days_with_Sales']]

# Need to encode categorical variables because most machine learning libraries requires numeric input.
# dataframe.select_dtypes(include=[column type]).columns
categorical_columns = X.select_dtypes(include=['object']).columns

# Apply one-hot encoding to categorical columns
X_encoded = pd.get_dummies(X
                           , columns=categorical_columns
                           , drop_first=True)




# target variable
y = df_model_compare['R_squared_Group_SimpleRegression']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded
                                                    , y
                                                    , test_size=0.2
                                                    , random_state=42)

# Train a decision tree classifier
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# Predict and evaluate
y_pred = tree.predict(X_test)
print(metrics.classification_report(y_test, y_pred))

# Feature importance
importance = tree.feature_importances_
features = X_encoded.columns

# Create dataframe of variable importance
df_dt_imp = pd.DataFrame({'Feature': features
                          ,'Importance': importance})

##### Prepare features & Train model


##### Visualization

### Visualize the tree

### Saveout png after pruned the tree 

# Optional: Prune the tree with different thresholds and visualize
tree_pruned = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_pruned.fit(X_train, y_train)

# # Figure size
# plt.figure(figsize=(20,10),dpi=300)  # Increase dpi for higher resolution)

# # Plot function
# plot_tree(tree_pruned
#           , filled=True
#           , feature_names=X_encoded.columns
#           , class_names=y.unique())

# # Save plot
# pic_dt_R_group = fld_analysis + "DecisionTree_R_Group.png"
# print(pic_dt_R_group)
# plt.savefig(pic_dt_R_group
#             ,bbox_inches='tight')

# # Show plot
# plt.show()


# Export as dot file and convert to SVG(Scalable Vector Graphics)
dot_data = export_graphviz(tree_pruned
                           , out_file=None, 
                           feature_names=X_encoded.columns,  
                           class_names=y.unique(),
                           filled=True)

# Use graphviz to render and save as SVG
graph = graphviz.Source(dot_data)

pic_dt_R_group = fld_analysis + "DecisionTree_R_Group.svg"
print(pic_dt_R_group)

graph.render(pic_dt_R_group)

########## End of Analysis of linearity & low R-square products ##########

