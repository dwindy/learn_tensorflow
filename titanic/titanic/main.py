import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# matplotlib inline

data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

print data_train.sample(3) #random select

#sns.barplot(x="Embarked", y="Survived", hue="Sex", data=data_train)
#plt.show()

# sns.pointplot(x="Embarked", y="Survived", hue="Sex", data=data_train,palette={"male":"blue","female":"pink"},markers=["*","o"],linestyles=["-","--"])
# plt.show()

def simplify_ages(data):
        data.Age = data.Age.fillna(-0.5)


