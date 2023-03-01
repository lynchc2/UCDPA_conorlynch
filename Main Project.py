import numpy as np
import pandas as pd
#pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt
import seaborn as sns
#import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


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



# Convert and Scale Data
scaler = LabelEncoder()
for column in ['type_of_meal_plan', 'room_type_reserved','market_segment_type', 'booking_status']:
    data[column] = scaler.fit_transform(data[column])

y = data['booking_status']
X = data.drop(['booking_status'],axis= 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X)



# Correlation between different columns
plt.figure(figsize=(10,7))
sns.heatmap(data.corr().round(2), annot=True, cmap='Blues', xticklabels=1, yticklabels=1)
#sns.set_xticklabels(sns.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()


# Correlation to booking status
plt.figure(figsize=(10,7))
data.corr()['booking_status'].sort_values(ascending=False).plot(kind='bar')
plt.xticks(rotation=45, ha='right')
plt.show()


# Examining lead time & price relation with cancelling
plt.figure(figsize=(10,7))
sns.scatterplot(data=data, x='avg_price_per_room', y='lead_time', hue='booking_status')
plt.show()



# Train Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=21, stratify=y)



# Decision Tree Classifier
dcc = DecisionTreeClassifier(random_state=21)
dcc.fit(X_train, y_train)
y_pred = dcc.predict(X_test)

p_score = precision_score(y_test, y_pred)
acc_score = accuracy_score(y_test, y_pred)
r_score = recall_score(y_test, y_pred)
final_results = pd.DataFrame([['Decision Tree Classifier', acc_score, p_score, r_score]],
                             columns=['Model', 'Accuracy Score', 'Precision Score', 'Recall Score'])

c_matrix = confusion_matrix(y_test, y_pred)
#print (c_matrix)


# Random Forest Classifier
rfc = RandomForestClassifier(random_state=21)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

p_score = precision_score(y_test, y_pred)
acc_score = accuracy_score(y_test, y_pred)
r_score = recall_score(y_test, y_pred)
final_results1 = pd.DataFrame([['Random Forest Classifier', acc_score, p_score, r_score]],
                              columns=['Model', 'Accuracy Score', 'Precision Score', 'Recall Score'])
final_results1 = final_results.append(final_results1)

c_matrix = confusion_matrix(y_test, y_pred)
#print (c_matrix)


# XGB Classifier
xgb = XGBClassifier(random_state=21)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

p_score = precision_score(y_test, y_pred)  # number of true positives over all positive predictions, high means low false +ve rate
acc_score = accuracy_score(y_test, y_pred)
r_score = recall_score(y_test, y_pred)
final_results2 = pd.DataFrame([['XG Boost Classifier', acc_score, p_score, r_score]],
                              columns=['Model', 'Accuracy Score', 'Precision Score', 'Recall Score'])
final_results2 = final_results1.append(final_results2)
print (final_results2)

c_matrix = confusion_matrix(y_test, y_pred)
#print (c_matrix)
