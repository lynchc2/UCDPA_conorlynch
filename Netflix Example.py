import numpy as np
import pandas as pd
import re
from statistics import mean

df = pd.read_csv("C:/Users/lynchc2/OneDrive - Paddy Power Betfair/Conor Lynch/UCD Course/Netflix Titles.csv")

print (df.describe())
print (df.head())
print (df.isna().sum())
print (df.info())

#Replacing nulls
df['imdb_score'] = df['imdb_score'].fillna(df['imdb_score'].median())
df['title'].replace(np.nan, 'No Data',inplace  = True)

#Drop duplicate rows
df.drop_duplicates(inplace= True)
print (df.isna().sum().sort_values(ascending=False))

#Drop 1st column which is just an id
df=df.drop('id',axis=1)
print (df.head())

#Regex and function to find patterns
regex = (r'\d+')
title_list = df['description'].astype(str)

for i in title_list:
    #Find all matches of regex in each title
    cowboy = re.findall(regex, i)

    print ("Numbers contrained in this description{}".format(cowboy))









