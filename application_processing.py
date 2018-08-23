import pandas as pd
import numpy as np

# load
df_train = pd.read_csv('data/application_train.csv')
df_test = pd.read_csv('data/application_test.csv')

# lower column names
df_train.columns = [col.lower() for col in df_train.columns]
df_test.columns = [col.lower() for col in df_test.columns]


# Drop

# These columns contain normalized information about where the client lives.
# Each column is missing about 45-70% of the data.
# Could do something with this later if you care to.
df_train.drop(df_train.columns[range(44, 91)], inplace=True, axis=1)
df_test.drop(df_test.columns[range(43, 90)], inplace=True, axis=1)

# these columns were not viewed as important during EDA
drop_cols = [
    'weekday_appr_process_start',
    'amt_req_credit_bureau_hour',
    'amt_req_credit_bureau_week',
    'amt_req_credit_bureau_mon',
    'amt_req_credit_bureau_year',
]

for col in drop_cols:
    df_train.drop(col, inplace=True, axis=1)
    df_test.drop(col, inplace=True, axis=1)


# Bin

# Impute gender, assuming female (2 to 1 ratio of F to M in data)
idxs_train = df_train[df_train['code_gender'] == 'XNA'].index
idxs_test = df_test[df_test['code_gender'] == 'XNA'].index

for idx in idxs_train:
    df_train.at[idx, 'code_gender'] = 'F'

for idx in idxs_test:
    df_test.at[idx, 'code_gender'] = 'F'

# Change this from code_gender to is_male
m_or_f = lambda x: 1 if x == 'M' else 0
df_train['code_gender'] = df_train['code_gender'].apply(m_or_f)
df_train.rename(columns={'code_gender': 'is_male'}, inplace=True)
