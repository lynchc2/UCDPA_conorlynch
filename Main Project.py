import numpy as np
import pandas as pd
#pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

data = pd.read_csv("C:/Users/lynchc2/OneDrive - Paddy Power Betfair/Conor Lynch/UCD Course/Hotel Reservations.csv")

print (data.describe())
print (data.head())
print (data.info())


#Checking for NULLs
print (data.isna().sum())


#Drop duplicate rows
data.drop_duplicates(inplace= True)


#Dropping Booking_Id column
data.drop('Booking_ID',axis=1,inplace=True)


#Examining distribution of the measures

def plot_graphs (column):
    """Returns a bar graph of column types counted"""
    sns.countplot(x=column, data=data, hue='booking_status')
    plt.ylabel('Number of Reservations')
    plt.show()

#Dropping some columns as it doesnt make sense in plots (continuos?)
plot_columns = data.drop(['booking_status','lead_time','arrival_date','no_of_previous_bookings_not_canceled','avg_price_per_room'],axis=1)

for i in plot_columns:
    plot_graphs(i)


def plot_graphs2 (column):
    """Returns a histogram of column types counted"""
    sns.histplot(x=column, data=data, hue='booking_status')
    plt.ylabel('Number of Reservations')
    plt.show()

#Only looking at columns dropped from previous plots
plot_columns2 = data[['lead_time','avg_price_per_room','arrival_date','no_of_previous_bookings_not_canceled']]

for i in plot_columns2:
    plot_graphs2(i)


#Correlation between different columns
#plt = px.imshow(data.corr(), color_continuous_scale="Reds")
#plt.update_layout(height=800)
#plt.show()

#df_corr_bar = data.corr().booking_status.sort_values()[:-1]
#fig = px.bar(df_corr_bar, orientation="h", color_discrete_sequence=["#AEC6CF"])
#fig.update_layout(showlegend=False)
#fig.show()
