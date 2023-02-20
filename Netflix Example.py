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
regex1 = (r'Cowboy|cowboy')
regex2 = (r'Football|football')
description_list = df['description'].astype(str)

for i in description_list:
    #Find all matches of regex in each description
    cowboy = re.findall(regex1, i)
    football = re.findall(regex2, i)

    if cowboy:
        print ('Cowboy Film')

    if football:
        print ('Football Film')

    else:
        print ('Other')

a = (cowboy, football)

print (a)

#Iterators
titles = df['title']
myiter = iter(titles)

print (next(myiter))
print (next(myiter))
print (next(myiter))

#Merge Dataframe














