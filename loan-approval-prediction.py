
# Loan Approval Prediction
# ## 1. Import Packages & Data

# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# Read data
df = pd.read_csv("train_u6lujuX_CVtuZ9i (1).csv")

# preview data
df.head()


# ## 2. Data Quality & Missing Value Assesment
# Preview data information
df.info()
description = df.describe()
# Check missing values
df.isnull().sum()


# ### Gender - Missing Values
# percent of missing "Gender" 
print('Percent of missing "Gender" records is %.2f%%' % ((df['Gender'].isnull().sum() / df.shape[0])*100))
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

#the loan ammount is skewed and have outliers
df['LoanAmount'].median()
df['LoanAmount'].mode()
df['LoanAmount'].mean()
sns.boxplot(y='LoanAmount',data=df)
sns.histplot(data=df,x='LoanAmount', palette = 'Set2')



# ## 3. Final Adjustments to Data
# * If "Gender" is missing = Male (mode).
# * If "Married" is missing = yes (mode).
# * If "Dependents" is missing = 0 (mode).
# * If "Self_Employed" is missing = no (mode).
# * If "LoanAmount" is missing = median of data. (it`s a neumiric data, mode doesn`t make sense)
# * If "Loan_Amount_Term" is missing = 360 (mode).
# * If "Credit_History" is missing = 1.0 (mode).

train_data = df.copy()
train_data['Gender'].fillna(train_data['Gender'].mode()[0], inplace=True)  
train_data['Married'].fillna(train_data['Married'].mode()[0], inplace=True)
train_data['Dependents'].fillna(train_data['Dependents'].mode()[0], inplace=True)
train_data['Self_Employed'].fillna(train_data['Self_Employed'].mode()[0], inplace=True)
train_data["LoanAmount"].fillna(train_data["LoanAmount"].median(), inplace=True)
train_data['Loan_Amount_Term'].fillna(train_data['Loan_Amount_Term'].mode()[0], inplace=True)
train_data['Credit_History'].fillna(train_data['Credit_History'].mode()[0], inplace=True)


#x = train_data['Married'].mode() //returns series of shape (1,)
#x_var = train_data['Married'].mode()[0] //returns the mode

#Check missing values
train_data.isnull().sum()
train_data


#transformation
"""hot_encoder = OneHotEncoder(handle_unknown='error')
transformed_train_data = hot_encoder.fit_transform(train_data[['Gender','Married','Dependents','Education','Self_Employed','Property_Area']])
transformed_train_data= transformed_train_data.toarray()
print(hot_encoder.categories_)
in this problem data is ordinal """ 

#transformation and Convert categorical object data type to int64

gender_stat = {"Female": 1, "Male": 2}
yes_no_stat = {'No' : 1,'Yes' : 2}
dependents_stat = {'0':0,'1':1,'2':2,'3+':3}
education_stat = {'Not Graduate' : 1, 'Graduate' : 2}
property_stat = {'Semiurban' : 0, 'Urban' : 1,'Rural' : 2}

train_data['Gender'] = train_data['Gender'].replace(gender_stat)
train_data['Married'] = train_data['Married'].replace(yes_no_stat)
train_data['Dependents'] = train_data['Dependents'].replace(dependents_stat)
train_data['Education'] = train_data['Education'].replace(education_stat)
train_data['Self_Employed'] = train_data['Self_Employed'].replace(yes_no_stat)
train_data['Property_Area'] = train_data['Property_Area'].replace(property_stat)



#scale numirical data 
sns.countplot(x='Loan_Amount_Term',data=df,palette='Set3')
"""loan amount term is numerical data not following the normal distribution """
#minimax scaler
#for numiric
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(train_data.iloc[:,6:10])
normData = pd.DataFrame(min_max_scaler.transform(train_data.iloc[:,6:10]), index=train_data.index, columns=train_data.iloc[:,6:10].columns)
train_data.iloc[:,6:10] = normData

#for property area and dependents
prop_depend_scaler = MinMaxScaler()
prop_depend_scaler.fit(train_data.loc[:,['Dependents','Property_Area']])
prop_depend_norm = pd.DataFrame(prop_depend_scaler.transform(train_data.loc[:,['Dependents','Property_Area']]), index=train_data.loc[:,['Dependents','Property_Area']].index, columns=train_data.loc[:,['Dependents','Property_Area']].columns)
train_data.loc[:,['Dependents','Property_Area']] = prop_depend_norm


#Preview data information
df.info()
df.isnull().sum()

train_data.info()
train_data.isnull().sum()
data_description_after_preprocessing = train_data.describe()

# ######data is clean and pre processed#########

# split data
x = train_data.iloc[:, 1:12]
y = train_data.iloc[:, 12]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1234)

# 4.Making Prediction

# Separate feature and target
# models and it`s evaluation
scores = []
classifier = ('Gradient Boosting' , 'Random Forest' ,'Decision Tree' , 'K-Nearest Neighbor' , 'SVM' ,'XGBoost','LogisticRegression')
y_pos = np.arange(len(classifier))
#GradientBoostingClassifier
GBC = GradientBoostingClassifier()
GBC.fit(X_train, y_train)
expected_y  = y_test
predicted_y = GBC.predict(X_test)
GBC_accuracy_score = accuracy_score(expected_y, predicted_y)*100
scores.append(GBC_accuracy_score)
print('The accuration of classification is %.2f%%' % (GBC_accuracy_score))

#RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=10)
RFC.fit(X_train, y_train)
expected_y  = y_test
predicted_y = RFC.predict(X_test)
RFC_accuracy_score = accuracy_score(expected_y, predicted_y)*100
scores.append(RFC_accuracy_score)
print('The accuration of classification is %.2f%%' %(RFC_accuracy_score))


#DecisionTreeClassifier
DTC = DecisionTreeClassifier()
DTC.fit(X_train, y_train)
expected_y  = y_test
predicted_y = DTC.predict(X_test)
DTC_accuracy_score = accuracy_score(expected_y, predicted_y)*100
scores.append(DTC_accuracy_score)
print('The accuration of classification is %.2f%%' %(accuracy_score(expected_y, predicted_y)*100))


#KNeighborsClassifier
KNN = KNeighborsClassifier()
KNN.fit(X_train, y_train)
expected_y  = y_test
predicted_y = KNN.predict(X_test)
KNN_accuracy_score = accuracy_score(expected_y, predicted_y)*100
scores.append(KNN_accuracy_score)
print('The accuration of classification is %.2f%%' %(KNN_accuracy_score))



#LinearSVC
SVM = svm.LinearSVC(max_iter=5000)
SVM.fit(X_train, y_train)
expected_y  = y_test
predicted_y = SVM.predict(X_test)
SVM_accuracy_score = accuracy_score(expected_y, predicted_y)*100
scores.append(SVM_accuracy_score)
print('The accuration of classification is %.2f%%' %(SVM_accuracy_score))


#XGBClassifier
XGB = xgb.XGBClassifier()
XGB.fit(X_train, y_train)
expected_y  = y_test
predicted_y = XGB.predict(X_test)
XGB_accuracy_score = accuracy_score(expected_y, predicted_y)*100
scores.append(XGB_accuracy_score)
print('The accuration of classification is %.2f%%' %(XGB_accuracy_score))



#LogisticRegression
LR = LogisticRegression()
LR.fit(X_train, y_train)
expected_y  = y_test
predicted_y = LR.predict(X_test)
LR_accuracy_score = accuracy_score(expected_y, predicted_y)*100
scores.append(LR_accuracy_score)
print('The accuration of classification is %.2f%%' %(LR_accuracy_score))





# ## 5. Result

# The result is Gradient Boosting Classifier have the highest score from other classification algorithm. These result are similar to my previous works.


plt.barh(y_pos, scores, align='center', alpha=0.5)
plt.yticks(y_pos, classifier)
plt.xlabel('Score')
plt.title('Classification Performance')
plt.show()


# ## Reference
# 1. J. Heo and J. Y. Yang, "AdaBoost Based Bankruptcy Forecasting of Korean Construction Company," Applied Soft Computing, vol. 24, pp. 494-499, 2014.
# 2. C.-F. Tsai, "Feature Selection in Bankruptcy Prediction," Knowledge Based System, pp. 120-127, 2009.





