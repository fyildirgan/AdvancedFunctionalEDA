# 2. Kategorik Degisken Analizi(Analysis of Categorical Variables)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
#print(df.head())

#print(df["embarked"].value_counts())
#print(df["sex"].unique())
#print(df["class"].nunique())

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
#print(cat_cols)

num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]
#print(num_but_cat)

cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
#print(cat_but_car)

cat_cols = cat_cols + num_but_cat

cat_cols = [col for col in cat_cols if col not in cat_but_car]
df[cat_cols]

df[cat_cols].nunique()
[col for col in df.columns if col not in cat_cols]

df["survived"].value_counts()
100 * df["survived"].value_counts() / len(df)


def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                         "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("###########################################")



print(cat_summary(df,'sex'))




# Butun kategorik degiskenleri cagirma:
for col in cat_cols:
    cat_summary(df, col)
print(cat_cols)


