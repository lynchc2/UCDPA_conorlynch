import numpy as np
import pandas as pd
import re
from statistics import mean

df = pd.read_csv("C:/Users/lynchc2/OneDrive - Paddy Power Betfair/Conor Lynch/UCD Course/Netflix Titles.csv")

print (df.describe())
print (df.head())
print (df.isna().sum())

#Replacing nulls
df['imdb_score'] = df['imdb_score'].fillna(df['imdb_score'].median())
df['title'].replace(np.nan, 'No Data',inplace  = True)

#Drop duplicate rows
df.drop_duplicates(inplace= True)

print (df.isna().sum())




