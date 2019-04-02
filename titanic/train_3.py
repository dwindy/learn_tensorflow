##
#ref:https://nbviewer.jupyter.org/github/HanXiaoyang/Kaggle_Titanic/blob/master/Titanic.ipynb
#ref:https://github.com/HanXiaoyang/Kaggle_Titanic
##
import pandas as pd   
import numpy as np  
from pandas import Series, DataFrame
import matplotlib.pyplot as plt 

data_train = pd.read_csv("train.csv")
print data_train.columns

print data_train.info()

print data_train.describe()

fig = plt.figure(0)
fig.set(alpha = 0.2)

plt.subplot2grid((2,3),(0,0))
data_train.Survived.value_counts().plot(kind='bar')
plt.title(u"situation(1 for survied)")
plt.ylabel(u"number")

plt.subplot2grid((2,3), (0,1))
data_train.Pclass.value_counts().plot(kind='bar')
plt.title(u"ticket distribution")
plt.ylabel(u"number")

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel(u"age")
#plt.grid(b=True, which='major', axis='y')
plt.title(u"survied situation")

plt.subplot2grid((2,3),(1,0), colspan=2)
data_train.Age[data_train.Pclass==1].plot(kind='kde') #kernel desnsity estimate
data_train.Age[data_train.Pclass==2].plot(kind='kde')
data_train.Age[data_train.Pclass==3].plot(kind='kde')
plt.xlabel(u"age")# plots an axis lable
plt.ylabel(u"density") 
plt.title(u"age over class")
plt.legend((u'1st', u'2ed',u'3rd'),loc='best') # sets our legend for our graph.

plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u"each dst")
plt.ylabel(u"number")  
#plt.show()

# #
# fig = plt.figure(1)
# fig.set(alpha=0.2)
# Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
# Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
# df=pd.DataFrame({u'survived':Survived_1, u'not survived':Survived_0})
# df.plot(kind='bar', stacked=True)
# plt.title(u"survived situation each class")
# plt.xlabel(u"class") 
# plt.ylabel(u"number") 
# plt.show()

from sklearn.ensemble import RandomForestRegressor

### fill null age by randomforestclassifier
def set_missing_ages(df):
    #extract exsist features
    age_df = df[['Age','Fare','Parch','SibSp','Pclass']]
    #dividing passenger into two groups by age
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    #every age col from all rows
    y = known_age[:,0]
    #rest cols from all rows
    X = known_age[:, 1:]

    #fit into randomforestregressor
    rfr = RandomForestRegressor(random_state = 0, n_estimators = 2000, n_jobs = -1)
    rfr.fit(X, y)

    #predict age with model upon
    predictAges = rfr.predict(unknown_age[:,1::])

    #fill null age with predict value
    df.loc[(df.Age.isnull()), 'Age'] = predictAges

    return df, rfr

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df

data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)
print data_train.describe()

###trans string into numbers
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')
df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

###scalling data
import sklearn.preprocessing as preprocessing
'''
Reshape your data either using array.reshape(-1, 1)
if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
'''
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'].values.reshape(-1, 1))
df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1), age_scale_param)
fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1, 1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1, 1), fare_scale_param)
df
#should we drop orin age and fare?

###extract feature, trans to numpy
from sklearn import linear_model
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()
y = train_np[:, 0]
X = train_np[:, 1:]

# fit into RandomForestRegressor
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X, y)
print clf


###same action in valid dataset
data_test = pd.read_csv("test.csv")
print data_test.describe()

data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0 #why set 0?

tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()

X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')


df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1, 1), age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1, 1),fare_scale_param)
df_test

###let's roll
test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("logistic_regression_predictions.csv", index=False)