import numpy as np
import pandas as pd
#pd.set_option('display.max_columns', None)

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix, accuracy_score


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

#for i in plot_columns:
#    plot_graphs(i)


def plot_graphs2 (column):
    """Returns a histogram of column types counted"""
    sns.histplot(x=column, data=data, hue='booking_status')
    plt.ylabel('Number of Reservations')
    plt.show()

#Only looking at columns dropped from previous plots
plot_columns2 = data[['lead_time','avg_price_per_room','arrival_date','no_of_previous_bookings_not_canceled']]

#for i in plot_columns2:
#    plot_graphs2(i)


#Correlation between different columns
#plt = px.imshow(data.corr(), color_continuous_scale="Reds")
#plt.update_layout(height=800)
#plt.show()

#df_corr_bar = data.corr().booking_status.sort_values()[:-1]
#fig = px.bar(df_corr_bar, orientation="h", color_discrete_sequence=["#AEC6CF"])
#fig.update_layout(showlegend=False)
#fig.show()

# Convert and Scale Data
scaler = LabelEncoder()
for column in ['type_of_meal_plan', 'room_type_reserved','market_segment_type', 'booking_status']:
    data[column] = scaler.fit_transform(data[column])

y = data['booking_status']
X = data.drop(['booking_status'],axis= 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=21, stratify=y)

# Decision Tree Classifier
dcc = DecisionTreeClassifier()
dcc.fit(X_train, y_train)
y_pred = dcc.predict(X_test)

p_score = precision_score(y_test, y_pred)
print("Precision Score:", p_score)
acc_score = accuracy_score(y_test, y_pred)
print("Accuracy Score:", acc_score)
c_matrix = confusion_matrix(y_test, y_pred)
print (c_matrix)

# Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

p_score = precision_score(y_test, y_pred)
print ("Precision Score:", p_score)
acc_score = accuracy_score(y_test, y_pred)
print ("Accuracy Score:", acc_score)
c_matrix = confusion_matrix(y_test, y_pred)
print (c_matrix)

# XGB Classifier
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

p_score = precision_score(y_test, y_pred)  # number of true positives over all positive predictions, high means low false +ve rate
print ("Precision Score:", p_score)
acc_score = accuracy_score(y_test, y_pred)
print ("Accuracy Score:", acc_score)
c_matrix = confusion_matrix(y_test, y_pred)
print (c_matrix)
# classification report to summarise multiple accuracy metrics

#Feature Importance
sns.set_style('whitegrid')
sns.set_context('poster')
sns.set_palette('colorblind')
sns.set(rc={'figure.figsize':(12,8)})
sns.barplot(x=rfc.feature_importances_, y=X.columns)
plt.title('FEATURE IMPORTANCE')
plt.show()
