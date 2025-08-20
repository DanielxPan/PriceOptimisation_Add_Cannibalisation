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
            """
            Train a demand model. Public-safe default: returns a constant-elasticity
            pseudo-model with elasticity=-1.0 and intercept auto-fitted from data.
            The returned object must have a .predict(df) method.
            """
            class _CEModel:
                def __init__(self, elasticity=-1.0):
                    self.elasticity = elasticity
                    self.y_mean = float(train_df[target].mean()) if len(train_df) else 0.0
                    self.p_mean = float(train_df['price'].mean()) if 'price' in train_df else 1.0

                def predict(self, df):
                    import numpy as np
                    p = df['price'].values if 'price' in df else self.p_mean
                    # Q = Q0 * (p/p0)^elasticity
                    base = self.y_mean if self.p_mean == 0 else self.y_mean
                    p0 = self.p_mean if self.p_mean else 1.0
                    return base * (p / p0) ** self.elasticity
            return _CEModel()

        @staticmethod
        def feasible_cost(product_row, cost_df):
            """
            Pick a usable cost for the product_row. Public-safe default:
            - use the minimum non-null cost column if present, else 0.0
            """
            import numpy as np
            if cost_df is None or len(cost_df)==0:
                return 0.0
            cols = [c for c in cost_df.columns if 'cost' in c.lower()]
            if not cols:
                return 0.0
            return float(np.nanmin(cost_df[cols].to_numpy()))

        @staticmethod
        def simulate_prices(model, grid, cost):
    return _strategy.simulate_prices(model, grid, cost)

Created on Fri Aug 30 10:17:15 2024

@author: danielp
"""

import pandas as pd
import datetime as dt
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import re



### Define function to sanitize string
def sanitize_string(string):
                        # ^ means not
    sanitized = re.sub(r'[^A-Za-z0-9\s]','',string)
    return sanitized



############### Block 1: Read raw data & Re-categorize ###############

##### Step 1: Read dataset
# File path
fld_data = r"os.path.join(DATA_DIR, '')"
file_daily_sales = fld_data + "DailyQTY_SelectedCategories_24Q1_+11.csv"
print(file_daily_sales)

# Read file
df_raw_all = pd.read_csv(file_daily_sales, header=0)
df_raw_all.dtypes



##### Step 2: Check date range
max_date = pd.to_datetime(df_raw_all['Date'],format='%Y-%m-%d').max()
min_date = pd.to_datetime(df_raw_all['Date'],format='%Y-%m-%d').min()



##### Step 3: Filter targeted categories
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



##### Step4: Create a grouping level for each category
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



##### Step5: Remove health food category
# # Requested by Harrison on 19th June
fil_health_food = df_raw_all['Calculation Group'] == 'HEALTH FOODS'
df_raw_all = df_raw_all[~fil_health_food]



##### Step6: Check Grouping result
df_check_group = df_raw_all[['Department','Space','Category','Calculation Group']].drop_duplicates()



##### Step7: Filter out store = na
df_raw_all = df_raw_all[~df_raw_all['Store'].isna()]



### Stats1: Num of products
num_products_raw = df_raw_all['Product'].nunique()

############### End of Block 1: Read raw data & Re-categorize ###############





########## Block 2: Remove products without price change ###############

##### Step1: Create filter- Filter by products without price change
# Function to check if there is only one unique price in the group
def has_price_change(x):
    return x.nunique() > 1



##### Step2: Apply filter
df_raw_all = df_raw_all.groupby(['Store', 'Product']).filter(lambda x: has_price_change(x['Price']))



### Stats2: Num of products after filter applied
num_products_raw_price_change = df_raw_all['Product'].nunique()

########## End of Block 2: Remove product with few data points ###############





########## Block 3: Remove outliers needs to be before fill na ###############

##### Step1: build function to find bounds
def find_bounds(df,columns):
    # Get list
    for col in columns:
        # Find Q1, Q3, IQR
        Q1 = df[col].quantile(0.25) # 1st quartile
        Q3 = df[col].quantile(0.75) # 3rd quartile
        Q99 = df[col].quantile(0.99) # 99th quartile
        IQR = Q3 - Q1
        
        print(f"{col} Q1: {Q1}, Q3:{Q3}, IQR: {IQR} Q99:{Q99}")
        
        # Find lower & upper bounds
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        print(f"{col} Lower bound: {lower_bound}")
        print(f"{col} Upper bound: {upper_bound}")


##### Step2: Apply function
find_bounds(df_raw_all,['QTY', 'Price'])

### Result
# QTY Q1: 0.0, Q3:1.0, IQR: 1.0 Q99:7.0
# QTY Lower bound: -3.0
# QTY Upper bound: 4.0



##### Step3: Check by sales trend chart. QTY > 100 destroys the pattern of line chart
df_outliers = df_raw_all[df_raw_all['QTY']>100]



##### Step4: Apply function
# Find the max QTY of the group in df_outliers
df_outliers_max = df_outliers[['Store'
                               ,'Department'
                               ,'Space'
                               ,'Category'
                               ,'Product'
                               ,'Date'
                               ,'Price'
                               ,'Calculation Group']].drop_duplicates()



##### Step5: Apply filter
df_raw_all = df_raw_all[df_raw_all['QTY'] <= 100]



##### Step6: Group max without outliers
df_raw_all_max = df_raw_all.groupby(['Store', 'Calculation Group', 'Product'])['QTY'].max()

# Reset index
df_raw_all_max = df_raw_all_max.reset_index()



##### Step7: Find the max of outlier group
df_outliers_max = pd.merge(df_outliers_max
                           ,df_raw_all_max
                           ,how='left'
                           ,on = ['Store','Calculation Group','Product']
                            )



##### Step8: Append outlier back with max QTY
df_raw_all = pd.concat([df_raw_all,df_outliers_max]
                       ,axis=0
                       ,ignore_index=True)

# Sort DF
df_raw_all = df_raw_all.sort_values(by=['Store'
                                       ,'Calculation Group'
                                       ,'Product'
                                       ,'Date'
                                       ,'Price'
                                       ])

########## End of Block 3: Remove outliers ###############





#################### Block 4: Get price simulation of each product ###############

##### Step0: Check df
df_check = df_raw_all.head(60)



##### Step1: Filter columns and drop duplicates
df_raw_all = df_raw_all[['Store'
                         ,'Calculation Group'
                         ,'Product'
                        ,'Price'
                                       ]].drop_duplicates()



##### Step2: Creat empty list & price increment for loops

# Initiate empty list
dfs_simulation = []

# Define price increment
price_increment = 0.1
# price_increment = 0.2  # 0.2 is too loose



##### Step3: Create loops to conduct price simulation

##### Step3-1: Outer loop: filter by store
for store in df_raw_all['Store'].unique():
    # df for sales dataset
    df_raw_str = df_raw_all[df_raw_all['Store'] == store]
    print(store)
    
    ##### Step3-2: 1st inner loop: filter by Calculation Group within store
    for category in df_raw_str['Calculation Group'].unique():
        # df for sales dataset
        df_store_category = df_raw_str[df_raw_str['Calculation Group'] == category]
        print(category)

        ##### Step3-3: 2nd inner loop: filter by product within Calculation Group
        for product in df_store_category['Product'].unique():
            df_str_cat_prd = df_store_category[df_store_category["Product"] == product]
            print(product)
            
            # Filter max & min price to create the simulation price range
            min_price = df_str_cat_prd['Price'].min()
            max_price = df_str_cat_prd['Price'].max()
            
            # Round down the minimum price to 3 digits
            min_price = math.floor(min_price * 10) / 10
            
            # Round up the maximum price to 3 digits
            max_price = math.ceil(max_price * 10) / 10
            
            # price difference
            price_diff = round((max_price - min_price),2)
            
            # intervals
            invtervals = round( (price_diff / price_increment), 2 )
            
            # Add price series
            simulation_prices = pd.Series(
            [min_price + i * price_increment for i in range(int(invtervals + 1))]
                                        )

            # Make it a data frame
            simulation = pd.DataFrame({   'Store': [store] * len(simulation_prices)
                                            ,'Calculation Group': [category] * len(simulation_prices)
                                            ,'Product': [product] * len(simulation_prices)
                                            , 'Simulation_Prices': simulation_prices})
            
            # Append all dfs together
            dfs_simulation.append(simulation)
            
# Merge all DataFrames into a single DataFrame
df_simulation = pd.concat(dfs_simulation, ignore_index=True)

#################### End of Block 4: Get price simulation of each product ###############





#################### Block 5: Join cost df and remove # redacted proprietary negative-profit heuristics

##### Step1: Read cost book
fld_cost_data = r"os.path.join(DATA_DIR, '')"
file_cost_lowest = fld_cost_data + "Product_Cost_24Q1_Lowest.csv"
print(file_cost_lowest)

# Read cost book
df_cost = pd.read_csv(file_cost_lowest)



##### Step2: Join with simulation df
df_simulation = pd.merge(df_simulation
                         ,df_cost
                         ,how='left'
                         ,on=['Store'
                              ,'Product'])



##### Step3: Calculate profit per unit
df_simulation['Simulation_Profit'] = df_simulation['Simulation_Prices'] - df_simulation['Cost_Lowest']



##### Step4: Filter # redacted proprietary negative-profit heuristics
df_simulation = df_simulation[df_simulation['Simulation_Profit']>0]

#################### End of Block 5: Join cost df and remove # redacted proprietary negative-profit heuristics





#################### Block 6: Select model with higher R-Square ###############

##### Step1: Read model comparison table
fld_analysis = r"os.path.join(DATA_DIR, '')"

# Read file
file_model_cmp = fld_analysis + "ModelComparison_+11_Threshold_0.25_Check_VIF_P_Const.xlsx"
print(file_model_cmp)

df_model_cmp = pd.read_excel(file_model_cmp
                             ,sheet_name='Model_Comparison'
                             )



##### Step2: Filter product with OLS uplift
### Filter uplift > 0 products
df_model_cmp_uplift = df_model_cmp[df_model_cmp['Lift_by_Cannibalization']>0]

### Filter columns
df_model_cmp_uplift = df_model_cmp_uplift[['Store'
                                            ,'Calculation Group'
                                            ,'Product'
                                            ,'R-Squared_SimpleRegression'
                                            ,'R-Squared_OLS'
                                            ,'Lift_by_Cannibalization'
                                            ,'UpLift_IND']]



##### Step3: Read OSL table
df_model_ols = pd.read_excel(file_model_cmp
                             ,sheet_name='OLS'
                             )



##### Step4: Join 2 dfs together
df_model_ols = pd.merge(df_model_ols
                        ,df_model_cmp_uplift
                        ,how='inner'
                        ,on=['Store'
                             ,'Calculation Group'
                             ,'Product']
                        ,suffixes = ('','_Comparison')
                        )

############### End of Block 6: Select model with higher R-Square ###############





############### Block 7: Get simulation price & Calculate adjusted QTY, QTY adding cannibalization effect by adopting OLS ###############

######## Block 7-1: Get the price of the main product

##### Step1: Filter price only rows to get the coefficient of main product
df_model_ols_price = df_model_ols[[  'Store'
                                     ,'Calculation Group'
                                     ,'Product'
                                     ,'Variable'
                                     ,'Coefficient'
                                     ,'Intercept']].drop_duplicates()



##### Step2: Only filter price variables
df_model_ols_price = df_model_ols_price[df_model_ols_price['Variable'] == 'Price']



##### Step3: Join Simulation & OLS by products
df_model_ols_sml = pd.merge(df_model_ols_price
                            ,df_simulation
                            ,how='inner'
                            ,on=['Store'
                                 ,'Calculation Group'
                                 ,'Product']
                            )



##### Step4: Calculate QTY adjustment
# Create calculation- QTY Adjustment
df_model_ols_sml['QTY_Adjustment'] = df_model_ols_sml['Coefficient'] * df_model_ols_sml['Simulation_Prices']

######## End of Get the price of the main product





######## Block 7-2: Get the price of the cnnbl product

##### Step1: Filter price only rows to get the coefficient of main product
### Copy model ols sml
df_model_ols_cnnbl = df_model_ols[[  'Store'
                                     ,'Calculation Group'
                                     ,'Product'
                                     ,'Variable'
                                     ,'Coefficient']].drop_duplicates()



##### Step2: Only filter price variables
# Column list
list_var_remove = ['const','Price']
# Remove const price variable 
df_model_ols_cnnbl = df_model_ols_cnnbl[~(df_model_ols_cnnbl['Variable'].isin(list_var_remove))]



##### Step3: Join simulation on
df_model_ols_cnnbl = pd.merge(  df_model_ols_cnnbl
                                ,df_simulation
                                ,how='inner'
                                ,left_on=['Store'
                                         ,'Calculation Group'
                                         ,'Variable']
                                ,right_on=['Store'
                                         ,'Calculation Group'
                                         ,'Product']
                                ,suffixes = ('','_Cannibalized')
                                )



##### Step4: Remove duplication column
df_model_ols_cnnbl = df_model_ols_cnnbl.drop(columns=['Product_Cannibalized'])



##### Step5: Calculate QTY adjustment
# Create calculation- Predicted 
df_model_ols_cnnbl['QTY_Adjustment'] = df_model_ols_cnnbl['Coefficient'] * df_model_ols_cnnbl['Simulation_Prices']

######## End of Block 7-2: Get the price of the cnnbl product 



############### End of Block 7: Get simulation price & Calculate adjusted QTY, predicted QTY adding cannibalization effect by adopting OLS ###############





############### Block 8: Get predicted QTY of simple regression ###############

##### Step0: Prepare simple regression for cnbl product QTY

# list of columns
list_col_simple_reg = ['Store'
                       ,'Calculation Group'
                       ,'Product'
                       ,'Intercept_SimpleRegression'
                       ,'Coefficient_SimpleRegression'
                        ]

# Apply filter cols
df_simple_reg = df_model_cmp[list_col_simple_reg]



##### Step1: Join with simulation dfs
df_simple_reg_sml = pd.merge(df_simple_reg
                            ,df_simulation
                            ,how='inner'
                            ,on=['Store'
                                 ,'Calculation Group'
                                 ,'Product']
                            )



##### Step2: Calculate predicted QTY
df_simple_reg_sml['QTY_Predicted'] = (
    df_simple_reg_sml['Intercept_SimpleRegression']
                                      + 
    df_simple_reg_sml['Coefficient_SimpleRegression'] * df_simple_reg_sml['Simulation_Prices']
    )



##### Step3: change negative QTY as 0
df_simple_reg_sml['QTY_Predicted'] = np.where(df_simple_reg_sml['QTY_Predicted']<0,0,df_simple_reg_sml['QTY_Predicted'])



##### Step4: Calculate profit
df_simple_reg_sml['Profit_Predicted'] = df_simple_reg_sml['QTY_Predicted'] * df_simple_reg_sml['Simulation_Profit']

# Check samples
df_check_reg = df_simple_reg_sml.head(150)



##### Step5: only filter usable columns
df_simple_reg_sml = df_simple_reg_sml[['Store'
                                     ,'Calculation Group'
                                     ,'Product'
                                     ,'Simulation_Prices'
                                     ,'QTY_Predicted'
                                     ,'Profit_Predicted']].drop_duplicates()



##### Step6: Get max profit points

# Find the index of the highest profit point
idx = df_simple_reg_sml.groupby(['Store'
                                 ,'Calculation Group'
                                 # .idxmax() finds the index of the row with the maximum
                                 ,'Product'])['Profit_Predicted'].idxmax()

# Apply filter .loc
df_simple_reg_sml_max = df_simple_reg_sml.loc[idx]

############### End of Block 8: Get predicted QTY of simple regression ###############





################### Block 9: Filter testing stores and categories ###################

##### Step1: Create filter lists

list_sample_str = ['Country Store'
                  ,'Metro Store'
                  ,'Metro Discount']


list_test_category = ['CHOCOLATES'
                      ,'BEVERAGES'
                      ,'BISCUITS'
                      ,'CONFECTIONERY']



##### Step2: Apply filters for OLS regression model
fil_sample_str_sml = df_model_ols_sml['Store'].isin(list_sample_str)
fil_test_prd_sml = df_model_ols_sml['Calculation Group'].isin(list_test_category)

df_model_ols_sml = df_model_ols_sml[fil_sample_str_sml & fil_test_prd_sml]



##### Step3: Apply filters for cannibalisation
fil_sample_str_cnnbl = df_model_ols_cnnbl['Store'].isin(list_sample_str)
fil_test_prd_cnnbl = df_model_ols_cnnbl['Calculation Group'].isin(list_test_category)


df_model_ols_cnnbl = df_model_ols_cnnbl[fil_sample_str_cnnbl & fil_test_prd_cnnbl]



##### Step4: Apply filters for simple regression model
fil_sample_str_simple = df_simple_reg_sml['Store'].isin(list_sample_str)
fil_test_prd_simple = df_simple_reg_sml['Calculation Group'].isin(list_test_category)

df_simple_reg_sml = df_simple_reg_sml[fil_sample_str_simple & fil_test_prd_simple]


################### End of Block 9: Filter testing stores and categories ###################





################### Block 10: Merge & Save out result ###################

##### Step1: Initiate lists
list_dfs_model_cnbl = []
list_dfs_model_profit_compare = []

### Step1: Read reference talbe
file_ref_store = r"REF_STORE_CSV"
df_store_ref = pd.read_csv(file_ref_store)


### Step2: For loop to subset dfs
# Subset by store
for store in df_model_ols_sml['Store'].unique():
    
    # df for cnbl model- main product part & intercept
    df_model_ols_sml_str = df_model_ols_sml[df_model_ols_sml['Store'] == store]
    
    # df for cnbl model- cnbl product & coefficient
    df_model_ols_cnnbl_str = df_model_ols_cnnbl[df_model_ols_cnnbl['Store'] == store]
    
    # df for simple regression
    df_simple_reg_sml_str = df_simple_reg_sml[df_simple_reg_sml['Store'] == store]
    
    print(f"Store: {store}")
    
    # Get unique category df
    for category in df_model_ols_sml_str['Calculation Group'].unique():
        
        # df for cnbl model- main product part & intercept
        df_model_ols_sml_str_cat = df_model_ols_sml_str[df_model_ols_sml_str['Calculation Group'] == category]
        
        # df for cnbl model- cnbl product & coefficient
        df_model_ols_cnnbl_str_cat = df_model_ols_cnnbl_str[df_model_ols_cnnbl_str['Calculation Group'] == category]
        
        # df for simple regression
        df_simple_reg_sml_str_cat = df_simple_reg_sml_str[df_simple_reg_sml_str['Calculation Group'] == category]

        print(f"Category: {category}")

        # Get unique products df
        # for product in list_test_prd: # For test
        for product in df_model_ols_sml_str_cat['Product'].unique():
            
            # Subset by product- main product
            df_model_ols_sml_str_cat_prd = df_model_ols_sml_str_cat[df_model_ols_sml_str_cat["Product"] == product]
            ### Calculate rows of df
            rows_prod = len(df_model_ols_sml_str_cat_prd)
            
            # Subset by product- cnnbl product
            df_model_ols_cnnbl_str_cat_prd = df_model_ols_cnnbl_str_cat[df_model_ols_cnnbl_str_cat["Product"] == product]
            print(f"Product: {product}")
            print(f"Rows main product: {rows_prod}")
            
            
            
            ###### Get max profit from simple regression to compare
            # Get the list of product combo
            list_product_comb = [product] + df_model_ols_cnnbl_str_cat_prd['Variable'].unique().tolist()
            
            # Filter simple reg max by list of products
            df_simple_reg_sml_max_prd = df_simple_reg_sml_max[df_simple_reg_sml_max['Product'].isin(list_product_comb)]
            
            ##### Pivot table
            df_simple_reg_sml_max_prd.dtypes
            ### Execute pivot table
            df_simple_reg_sml_max_prd_pvt = df_simple_reg_sml_max_prd.pivot_table(index=['Store'
                                                                                       ,'Calculation Group'
                                                                                       ]
                                                                                #Column values that will become the new columns
                                                                                ,columns='Product'
                                                                                #The values you want to fill in the new columns
                                                                                ,values=['Simulation_Prices'
                                                                                         ,'QTY_Predicted'
                                                                                         ,'Profit_Predicted'
                                                                                         ]).reset_index()
            
            ### Flatten the MultiIndex columns and rename them
                                                                            #  If the second level (col[1]) is empty, it only uses the first level (col[0])
                                                                                                    # .columns contains a MultiIndex
            df_simple_reg_sml_max_prd_pvt.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df_simple_reg_sml_max_prd_pvt.columns]
            
            ### Add product
            df_simple_reg_sml_max_prd_pvt['Product'] = product
            
            ### List of Profit predicted
            list_col_profit_predicted = [ col for col in df_simple_reg_sml_max_prd_pvt.columns if 'Profit_Predicted' in col ]
            
            ### Create Profit_Predicted_All by Regression
            df_simple_reg_sml_max_prd_pvt['Profit_Predicted_All'] = df_simple_reg_sml_max_prd_pvt[list_col_profit_predicted].sum(axis=1)
            
            ###### End of Get max profit from simple regression to compare
            
            
            ### Error handling
            try:
            
                ###### Get unique cnnbl products
                ### Need to reset the col before the next loop
                list_cnbl_prd_qty_adj = []
                list_cnbl_prd_price = []
                list_col_predicted_profit = []
                
                ### Start the loop of cnnbl
                for cnnbl in df_model_ols_cnnbl_str_cat_prd['Variable'].unique():
                    df_model_ols_cnnbl_str_cat_prd_var = df_model_ols_cnnbl_str_cat_prd[df_model_ols_cnnbl_str_cat_prd['Variable'] == cnnbl]
                    print(f"Cannbalized product: {cnnbl}")
                    
                    ### Calculate rows of df
                    rows_cnnbl = len(df_model_ols_cnnbl_str_cat_prd_var)
                    print(f"Rows of Cannbalized product: {rows_cnnbl}")
                    
                    ### Join with simple regression model to get Profit_Predicted
                    ### Need to add error handling
                    df_cnnbl_str_cat_prd_var_reg = pd.merge(df_model_ols_cnnbl_str_cat_prd_var
                                                            ,df_simple_reg_sml_str_cat
                                                            ,how='inner'
                                                            ,left_on=['Store'
                                                                        ,'Calculation Group'
                                                                        ,'Variable'
                                                                        ,'Simulation_Prices']
                                                            ,right_on=['Store'
                                                                        ,'Calculation Group'
                                                                        ,'Product'
                                                                        ,'Simulation_Prices']
                                                            ,suffixes=('','_Regression')
                                                            )
                    
                    
                    ### Append QTY addjustment columns into a list
                    col_qty_adj = "QTY_Adjustment_" + cnnbl
                    print(f"QTY_Adjustment: {col_qty_adj}")
                    list_cnbl_prd_qty_adj.append(col_qty_adj)
                    
                    ### Append Price Simulation columns into a list
                    col_prc_sml = "Simulation_Prices_" + cnnbl
                    list_cnbl_prd_price.append(col_prc_sml)
                    
                    ### Append Price Simulation columns into a list
                    col_predicted_profit = "Profit_Predicted_" + cnnbl
                    list_col_predicted_profit.append(col_predicted_profit)
                    
                    # Create Rename Dict 
                    dict_rename = {'Simulation_Prices': f"Simulation_Prices_{cnnbl}"
                                   ,'QTY_Adjustment': f"QTY_Adjustment_{cnnbl}"
                                   ,'Profit_Predicted': f"Profit_Predicted_{cnnbl}"}
                    
                    # Apply rename dict
                    df_cnnbl_str_cat_prd_var_reg.rename(columns=dict_rename
                                                              ,inplace=True)
                    
                    ### Drop duplicate column
                    df_cnnbl_str_cat_prd_var_reg.dtypes
                    df_cnnbl_str_cat_prd_var_reg.drop(columns=['Product_Regression'
                                                               ,'QTY_Predicted'
                                                               ,'Simulation_Profit'
                                                               ,'Cost_Lowest'
                                                               ,'Coefficient'
                                                               ,'Variable']
                                                      ,inplace=True)

                    # Join back main product dfs
                    df_model_ols_sml_str_cat_prd = pd.merge(df_model_ols_sml_str_cat_prd
                                                              ,df_cnnbl_str_cat_prd_var_reg
                                                              ,how='left'
                                                              ,on=['Store'
                                                                   ,'Calculation Group'
                                                                   ,'Product'])
            except MemoryError as mem_err:
                print(f"MemoryError encountered during merge for product {product}: {mem_err}")
                # Optionally: free up memory
                del df_cnnbl_str_cat_prd_var_reg
                del df_model_ols_sml_str_cat_prd
                import gc
                gc.collect()  # Explicit garbage collection to free up memory
                break  # Break out of the cnnbl loop and move to the next product
                
            except Exception as e:
                print(f"Error encountered during merge for product {product}: {e}")
                break  # Break out of the cnnbl loop and move to the next product
                
            ##### Out of cnnbl variable loop & process df by main product
                    
                    ### Calculation 1: Add all QTY adjustment
                    df_model_ols_sml_str_cat_prd['Predicted_QTY'] = (
                        df_model_ols_sml_str_cat_prd['Intercept'] + 
                        df_model_ols_sml_str_cat_prd['QTY_Adjustment'] +
                        df_model_ols_sml_str_cat_prd[list_cnbl_prd_qty_adj].sum(axis=1)
                                                                    )
                    
                    ### Calculation 1: Add all QTY adjustment
                    # Change negative QTY to 0
                    # Create filter
                    df_model_ols_sml_str_cat_prd['Predicted_QTY'] = np.where(
                        df_model_ols_sml_str_cat_prd['Predicted_QTY']<0
                        ,0
                        ,df_model_ols_sml_str_cat_prd['Predicted_QTY'])
                    
                    ### Calculation 2: Calculate Profit_Predicted by Predicted_QTY
                    df_model_ols_sml_str_cat_prd['Profit_Predicted'] = (
                        df_model_ols_sml_str_cat_prd['Predicted_QTY'] *
                        df_model_ols_sml_str_cat_prd['Simulation_Profit']
                        )
                    
                    ### Calculation 3: Sum profit of all products
                    df_model_ols_sml_str_cat_prd['Profit_Predicted_All'] = (
                        df_model_ols_sml_str_cat_prd['Profit_Predicted'] + 
                        df_model_ols_sml_str_cat_prd[list_col_predicted_profit].sum(axis=1)
                                                                    )
                    
                    # Filter by columns
                    list_col_final = [
                                       'Store'
                                      ,'Calculation Group'
                                      ,'Product'
                                      ,'Simulation_Prices'
                                      ,'Cost_Lowest'
                                      ,'Simulation_Profit'
                                      ] + list_cnbl_prd_price + list_col_predicted_profit + [
                                    'Predicted_QTY'
                                     ,'Profit_Predicted'
                                     ,'Profit_Predicted_All']
                    
                    # Apply column filter
                    df_model_ols_sml_str_cat_prd = df_model_ols_sml_str_cat_prd[list_col_final]
                    
                    # Find the max profit
                    max_profit = df_model_ols_sml_str_cat_prd['Profit_Predicted_All'].max()
                    df_model_cnbl_max_profit = df_model_ols_sml_str_cat_prd[df_model_ols_sml_str_cat_prd['Profit_Predicted_All'] == max_profit]

                # Join back the max profit from simple regression
                df_model_cnbl_reg_max_profit = pd.merge(
                                                        df_model_cnbl_max_profit
                                                        ,df_simple_reg_sml_max_prd_pvt
                                                        ,how='inner'
                                                        ,on=['Store'
                                                            ,'Calculation Group'
                                                            ,'Product']
                                                        ,suffixes=('_Cnbl','_Reg')
                                                        )
                
                # Add profit uplift
                df_model_cnbl_reg_max_profit['Profit_Predicted_All_Uplift'] = df_model_cnbl_reg_max_profit['Profit_Predicted_All_Cnbl'] - df_model_cnbl_reg_max_profit['Profit_Predicted_All_Reg']
                
                
                ##### Unpivot result to make it easier to analyze
                list_col_final = df_model_cnbl_reg_max_profit.columns.tolist()
                
                ##### Save result of every product
                
                ### Join store ref table to get store abbrv
                df_model_cnbl_reg_max_profit = pd.merge(df_model_cnbl_reg_max_profit
                                                        ,df_store_ref
                                                        ,how='inner'
                                                        ,left_on=['Store']
                                                        ,right_on=['StoreName'])
                
                ### Delete dup column
                df_model_cnbl_reg_max_profit.drop(columns=['StoreName']
                                                  ,inplace=True)
                
                ### Get store abbrv
                store_abbrv = ''.join(df_model_cnbl_reg_max_profit['StoreCodes'].unique().astype(str))
                
                ### Folder name
                fld_result_cnbl_product = r"os.path.join(DATA_DIR, '')"
                
                ### Sanitize product name
                product_sanitized = sanitize_string(product)
                
                ### Assemble file name
                file_name_max_profit = f"{store_abbrv}_{category}_{product_sanitized}.csv"
                
                ### Assemble file path 
                file_result_cnbl_product = fld_result_cnbl_product + file_name_max_profit
                print(file_result_cnbl_product)
                
                ## Save out result
                df_model_cnbl_reg_max_profit.to_csv(file_result_cnbl_product
                                                    ,index=False
                                                    ,header=True)

            ##### End of Save result of every product



            ##### Save result of all products            
            
            ### Common column
            df_model_cnbl_reg_max_profit.dtypes
            list_col_common = ['Store'
                               ,'Calculation Group'
                               ,'Product'
                               ,'Profit_Predicted'
                               ,'Profit_Predicted_All_Cnbl'
                               ,'Profit_Predicted_All_Reg'
                               ,'Profit_Predicted_All_Uplift']
            
            ### Filter by common columns
            df_model_cnbl_reg_max_profit_all_str = df_model_cnbl_reg_max_profit[list_col_common]

### Append all dfs into the list
list_dfs_model_profit_compare.append(df_model_cnbl_reg_max_profit_all_str)

### Concat all dfs
df_model_profit_compare = pd.concat(list_dfs_model_profit_compare)

### Create path and save out result
file_result_profit_compare = fld_result_cnbl_product + "Model_Profit_Compare.csv"
print(file_result_profit_compare)

df_model_profit_compare.to_csv(file_result_profit_compare
                               ,index=False
                               ,header=True)

##### End of Save result of every product



################### End of Block 10: Merge & Save out result ###################


