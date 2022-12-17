# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 22:21:20 2022

@author: babda
"""


#Loan Approval Prediction
# ## 1. Import Packages & Data

#Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import xgboost as xgb
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error,classification_report
#Read data
df = pd.read_csv("train_u6lujuX_CVtuZ9i (1).csv")

#preview data
df.head()

# ## 2. Data Quality & Missing Value Assesment


#Preview data information
df.info()

#Check missing values
df.isnull().sum()


# ### Gender - Missing Values
# percent of missing "Gender" 
print('Percent of missing "Gender" records is %.2f%%' %((df['Gender'].isnull().sum()/df.shape[0])*100))
# %s specifically is used to perform concatenation of strings together.
print("Number of people who take a loan group by gender :")
print(df['Gender'].value_counts())
#visuals
df['Gender'].value_counts().plot.bar(rot=0)
sns.countplot(x='Gender', data=df, palette = 'Set2')




# ### Married - Missing Values
# percent of missing "Married" 
print('Percent of missing "Married" records is %.2f%%' %((df['Married'].isnull().sum()/df.shape[0])*100))
print("Number of people who take a loan group by marital status :")
print(df['Married'].value_counts())
#visuals
df['Married'].value_counts().plot.bar(rot=0)
sns.countplot(x='Married', data=df, palette = 'Set2')



# ### Dependents- Missing Values
# percent of missing "Dependents" 
print('Percent of missing "Dependents" records is %.2f%%' %((df['Dependents'].isnull().sum()/df.shape[0])*100))
print("Number of people who take a loan group by dependents :")
print(df['Dependents'].value_counts())
#visuals
df['Dependents'].value_counts().plot.bar(rot=0)
sns.countplot(x='Dependents', data=df, palette = 'Set2')



# ### Education - Missing Values
# percent of missing "Education"
print('Percent of missing "Self_Employed" records is %.2f%%' %((df['Education'].isnull().sum()/df.shape[0])*100))
print("Number of people who take a loan group by Education :")
print(df['Education'].value_counts())
#visuals
df['Education'].value_counts().plot.bar(rot=0)
sns.countplot(x='Education', data=df, palette = 'Set2')



# ### Self Employed - Missing Values
# percent of missing "Self_Employed" 
print('Percent of missing "Self_Employed" records is %.2f%%' %((df['Self_Employed'].isnull().sum()/df.shape[0])*100))
print("Number of people who take a loan group by self employed :")
print(df['Self_Employed'].value_counts())
#visuals
df['Self_Employed'].value_counts().plot.bar(rot=0)
sns.countplot(x='Self_Employed', data=df, palette = 'Set2')


# ### Loan Amount - Missing Values
# percent of missing "LoanAmount" 
print('Percent of missing "LoanAmount" records is %.2f%%' %((df['LoanAmount'].isnull().sum()/df.shape[0])*100))

ax = df["LoanAmount"].hist(density=True, stacked=True, color='teal', alpha=0.6)
df["LoanAmount"].plot(kind='density', color='teal')
ax.set(xlabel='Loan Amount')
plt.show()


# ### Loan Amount Term - Missing Values
# percent of missing "Loan_Amount_Term" 
print('Percent of missing "Loan_Amount_Term" records is %.2f%%' %((df['Loan_Amount_Term'].isnull().sum()/df.shape[0])*100))

print("Number of people who take a loan group by loan amount term :")
print(df['Loan_Amount_Term'].value_counts())
sns.countplot(x='Loan_Amount_Term', data=df, palette = 'Set2')


# ### Credit History - Missing Values
# percent of missing "Credit_History" 
print('Percent of missing "Credit_History" records is %.2f%%' %((df['Credit_History'].isnull().sum()/df.shape[0])*100))
print("Number of people who take a loan group by credit history :")
print(df['Credit_History'].value_counts())

sns.countplot(x='Credit_History', data=df, palette = 'Set2')

#the loan ammount is skewed
df['LoanAmount'].median()
df['LoanAmount'].mode()
sns.boxplot(y='LoanAmount',data=df)
sns.histplot(data=df,x='LoanAmount', palette = 'Set2')



# ## 3. Final Adjustments to Data
# * If "Gender" is missing = Male (mode).
# * If "Married" is missing = yes (mode).
# * If "Dependents" is missing = 0 (mode).
# * If "Self_Employed" is missing = no (mode).
# * If "LoanAmount" is missing = mode of data. (although it is numerical data)
# * If "Loan_Amount_Term" is missing = 360 (mode).
# * If "Credit_History" is missing = 1.0 (mode).

train_data = df.copy()
train_data['Gender'].fillna(train_data['Gender'].mode()[0], inplace=True)  #the index of the maximum value
train_data['Married'].fillna(train_data['Married'].mode()[0], inplace=True)
train_data['Dependents'].fillna(train_data['Dependents'].mode()[0], inplace=True)
train_data['Self_Employed'].fillna(train_data['Self_Employed'].mode()[0], inplace=True)
train_data["LoanAmount"].fillna(train_data["LoanAmount"].mode()[0], inplace=True)
train_data['Loan_Amount_Term'].fillna(train_data['Loan_Amount_Term'].mode()[0], inplace=True)
train_data['Credit_History'].fillna(train_data['Credit_History'].mode()[0], inplace=True)




#Check missing values
train_data.isnull().sum()
train_data




#Convert some object data type to int64
gender_stat = {"Female": 0, "Male": 1}
yes_no_stat = {'No' : 0,'Yes' : 1}
dependents_stat = {'0':0,'1':1,'2':2,'3+':3}
education_stat = {'Not Graduate' : 0, 'Graduate' : 1}
property_stat = {'Semiurban' : 0, 'Urban' : 1,'Rural' : 2}

train_data['Gender'] = train_data['Gender'].replace(gender_stat)
train_data['Married'] = train_data['Married'].replace(yes_no_stat)
train_data['Dependents'] = train_data['Dependents'].replace(dependents_stat)
train_data['Education'] = train_data['Education'].replace(education_stat)
train_data['Self_Employed'] = train_data['Self_Employed'].replace(yes_no_stat)
train_data['Property_Area'] = train_data['Property_Area'].replace(property_stat)








x = train_data.iloc[:,1:12]
y = train_data.iloc[:,12]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1234)



#Preview data information
df.info()
df.isnull().sum()

train_data.info()
train_data.isnull().sum()




model = xgb.XGBClassifier()
model.fit(X_train, y_train)
print(); print(model)

expected_y  = y_test
predicted_y = model.predict(X_test)


yes_no_lable = {"Y":1,"N":0}
expected_y_num = #حول المتغير ل ارقام وجرب الاكيرسي وجرب كذا طريقة لقياس الكفاءة واعمل نورماليزيشن 
predicted_y_num
print(classification_report(expected_y, predicted_y))
print(confusion_matrix(expected_y, predicted_y))
print(accuracy_score(expected_y, predicted_y))
msqerr = mean_squared_error(expected_y, predicted_y)
# ## 5. Result
