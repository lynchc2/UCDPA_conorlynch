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

#Drop tmdb column as unsure what it is
df=df.drop('tmdb_score',axis=1)
print (df.head())

#Regex and function to find patterns, update to sentiment analysis??
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
titles1 = df['title']
myiter = iter(titles1)

print (next(myiter))
print (next(myiter))
print (next(myiter))

#Merge Dataframe

titles = pd.DataFrame(df['id'])
descriptions = pd.DataFrame(df[['id','title','type']])

new_df = titles.merge(descriptions, on='id')
print (new_df.head())

#NumPy picking out all PG films

arr = np.array(df['age_certification'])
print (np.where(arr == 'PG'))

arr2 = np.array(df['imdb_score'])
arr3 = np.array(df['title'])
filter_arr = []

for i in arr2:
    if i > 9:
        filter_arr.append(True)
    else:
        filter_arr.append(False)

new_arr = arr3[filter_arr]
print (filter_arr)
print (new_arr)

# Dictionaries and Lists










