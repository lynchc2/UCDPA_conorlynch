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


#Drop tmdb column
df=df.drop('tmdb_score',axis=1)


#Regex and function to find patterns
regex = (r'\d{4}')  ## just including this to demonstrate how I would of extracted titles containing years/ 4 digits
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


#Iterators
titles1 = df['title']
myiter = iter(titles1)

print (next(myiter))
print (next(myiter))
print (next(myiter))

#Extracting movies over 50 characters long
for element in titles1:
    if len(element) >= 50:
        print(element)

#Stopping a list once the movie character length drops below 10
for element in titles1:
    if len(element) >= 10:
        print(element)
    else:
        print ('!!! Next Movie Is Under 10 Characters !!!')
        break


#Merge Dataframe
titles = pd.DataFrame(df['id'])
descriptions = pd.DataFrame(df[['id','title','type']])

new_df = titles.merge(descriptions, on='id')
print (new_df.head())


#NumPy picking highly rated films, IMDB score > 9
arr2 = np.array(df['imdb_score'])
arr3 = np.array(df['title'])
filter_arr = []

for i in arr2:
    if i > 9:
        filter_arr.append(True)
    else:
        filter_arr.append(False)

new_arr = arr3[filter_arr]
print (new_arr)


# Dictionaries and Lists
list = df['title']
print (list[27])

newlist = [x for x in list if "world" in x]
print (newlist)








