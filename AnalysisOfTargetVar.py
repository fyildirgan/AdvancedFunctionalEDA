import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as pls
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: (dataframe[col_name].value_counts()),
                        "Ratio": 100 * (dataframe[col_name].value_counts()) / len(dataframe)}))
    print("###########################################")

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal degiskenlerin isimlerini verir.

    Parameters
    ----------
    dataframe: dataframe
        degisken isimleri alinmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan degiskenler icin sinif esik degeri
    car_th: int, float
        kategorik fakat kardinal degiskenleri icin sinif esik degeri

    Returns
    -------
    cat_cols: List
        Kategorik degisken listesi
    num_cols: List
         Numerik degisken listesi
    cat_but_car: List
         Kategorik gorunumlu kardinal degisken listesi

    Notes
    -------
    cat_cols + num_cols + cat_but_car = toplam degisken sayisi
    num_but_car cat_cols'un icerisinde.

    """
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int64", "float64"]]

    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.head()

df["survived"].value_counts()
cat_summary(df, "survived")

#Hedef Degiskenin Kategorik Degiskenler İle Analizi

df.groupby("sex")["survived"].mean()

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}))

target_summary_with_cat(df, "survived", "pclass")

for col in cat_cols:
    target_summary_with_cat(df, "survived", col)

# Hedef Degiskenin Sayisal Degiskenler ile Analizi

print(df.groupby("survived")["age"].mean())
print(df.groupby("survived").agg({"age":"mean"}))


def target_summary_with_num(dataframe, target, numarical_col):
    print(dataframe.groupby(target).agg({numarical_col: "mean"}), end="\n\n\n")

target_summary_with_num(df, "survived", "age")

for col in num_cols:
    target_summary_with_num(df, "survived", col)

