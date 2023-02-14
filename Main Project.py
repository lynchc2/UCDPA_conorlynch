import numpy as np
import pandas as pd

data = pd.read_csv("C:/Users/lynchc2/OneDrive - Paddy Power Betfair/Conor Lynch/UCD Course/Hotel Reservations.csv")

print (data.describe())
print (data.head())

print (data.isna().sum())

