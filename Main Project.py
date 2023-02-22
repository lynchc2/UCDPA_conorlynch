import numpy as np
import pandas as pd
#pd.set_option('display.max_columns', None)

data = pd.read_csv("C:/Users/lynchc2/OneDrive - Paddy Power Betfair/Conor Lynch/UCD Course/Hotel Reservations.csv")

print (data.describe())
print (data.head())
print (data.info())

#Checking for NULLs
print (data.isna().sum())

#Drop duplicate rows
data.drop_duplicates(inplace= True)



