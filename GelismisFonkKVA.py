# Gelismis Fonksiyonel Kesifci Veri Analizi
# Advanced Functional EDA
#1. Genel Resim
#2. Kategorik Degisken Analizi(Analysis of Categorical Variables)
#3. Sayisal Degisken Analizi(Analysis of Numerical Variables)
#4. Hedef Degisken Analizi(Analysis of Target Variable)
#5. Korelasyon Analizi(Analysis of Correlation)
# Amacı : Elimize buyuk yada kucuk veri geldiginde bu verileri olceklenebilir,
# yani fonksiyonel anlamda isleyebilmeyi hizli bir sekilde veriyle ilgili ic goruler
# edinebilmeyi amaclamaktadir.
# Hizli bir sekilde genel fonksiyonlar ile elimize gelen verileri analiz etmek

# 1. Genel Resim
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
#print(df.head())
def check_df(dataframe, head=5):
    print("###################### Shape ##################")
    print(dataframe.shape)
    print("###################### Types ##################")
    print(dataframe.dtypes)
    print("###################### Head ##################")
    print(dataframe.head(head))
    print("###################### Tail ##################")
    print(dataframe.tail(head))
    print("###################### NA ##################")
    print(dataframe.isnull().sum())
    print("###################### Quantiles ##################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

df = sns.load_dataset("flights")
check_df(df)
