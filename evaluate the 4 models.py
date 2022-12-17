# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 10:28:27 2022

@author: babda
"""

gender_stat = {"Female": 1, "Male": 2}
yes_no_stat = {'No' : 1,'Yes' : 2}
dependents_stat = {'0':0,'1':1,'2':2,'3+':3}
education_stat = {'Not Graduate' : 1, 'Graduate' : 2}
property_stat = {'Semiurban' : 0, 'Urban' : 1,'Rural' : 2}
without prop_depend_norm
LinearSVC The accuration of classification is 80.95%
KNeighborsClassifier The accuration of classification is 76.22%
DecisionTreeClassifier The accuration of classification is 72.31%
RandomForestClassifier The accuration of classification is 75.57%
GradientBoostingClassifier The accuration of classification is 77.36%
XGBClassifier 0.7804878048780488
LogisticRegression 0.8617886178861789











gender_stat = {"Female": 0, "Male": 1}
yes_no_stat = {'No' : 0,'Yes' : 1}
dependents_stat = {'0':0,'1':1,'2':2,'3+':3}
education_stat = {'Not Graduate' : 0, 'Graduate' : 1}
property_stat = {'Semiurban' : 0, 'Urban' : 1,'Rural' : 2}
without prop_depend_norm
LinearSVC The accuration of classification is 80.95%
KNeighborsClassifier The accuration of classification is 76.22%
DecisionTreeClassifier The accuration of classification is 71.34%
RandomForestClassifier The accuration of classification is 73.13%
GradientBoostingClassifier The accuration of classification is 77.36%
XGBClassifier  0.7804878048780488
LogisticRegression 0.8617886178861789





gender_stat = {"Female": 0, "Male": 1}
yes_no_stat = {'No' : 0,'Yes' : 1}
dependents_stat = {'0':0,'1':1,'2':2,'3+':3}
education_stat = {'Not Graduate' : 0, 'Graduate' : 1}
property_stat = {'Semiurban' : 0, 'Urban' : 1,'Rural' : 2}
with prop_depend_norm:
LinearSVC The accuration of classification is 80.95%
KNeighborsClassifier The accuration of classification is 78.50%
DecisionTreeClassifier The accuration of classification is 71.66%
RandomForestClassifier The accuration of classification is 75.89%
GradientBoostingClassifier The accuration of classification is 77.69%
XGBClassifier  0.7804878048780488
LogisticRegression 0.8617886178861789


with ordinal
LogisticRegression 0.8617886178861789
KNeighborsClassifier 78.50%
RandomForestClassifier 76.54%
GradientBoostingClassifier 77.52%
XGBClassifier 0.7804878048780488
