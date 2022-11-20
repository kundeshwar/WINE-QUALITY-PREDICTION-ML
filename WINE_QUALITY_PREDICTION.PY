#-----------------------------------------about dataset

#wine manufacturing company
#The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine. 
# For more details, consult the reference [Cortez et al., 2009].
#  Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).
#These datasets can be viewed as classification or regression tasks. 
# The classes are ordered and not balanced (e.g. there are much more normal wines than excellent or poor ones).

#-------------------------------------------work flow
#by using some parameter data we will predict quality of wine
#work flow
#collecting data 
#data-anlysis
#data pre-processing 
#train-test-split
#random forst data
#Below are some points that explain why we should use the Random Forest algorithm:
#It takes less training time as compared to other algorithms.
#It predicts output with high accuracy, even for the large dataset it runs efficiently.
#It can also maintain accuracy when a large proportion of data is missing.
#----------------------------------------import useful labrary
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

#---------------------------------data anlysis 
data = pd.read_csv("C:/Users/kunde/all vs code/ml prject/winequality-red.csv")
print(data.shape)
print(data.columns)
print(data.head(5))
print(data.describe())
print(data.info())
print(data.isnull().sum())#no need to edit null value becuse no null value is there
print(data["quality"].unique())
print(data["quality"].value_counts())
sns.catplot(x="quality", data=data, kind="count")
plt.show()
#this is graph which will give comparision of volatile acidity and quality of the wine
sns.barplot(x="quality", y="volatile acidity", data=data)
plt.show()
#this is graph which will give comparision of citric acidity and quality of the wine
sns.barplot(x="quality", y="citric acid", data=data)
plt.show()
#correlation
#positive correlation
#negitive correlation
correlation = data.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap="Blues")
plt.show()
#---------------------------------separte data

x = data.drop(columns="quality", axis=1)
y = data["quality"].apply(lambda x: 1 if x>=6 else 0)
print(y)

#---------------------------------train-test-split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=2, stratify=y, random_state=3)
print(x.shape, x_test.shape, x_train.shape)
print(y.shape, y_test.shape, y_train.shape)

#-------------------------------------------------use our model
model = RandomForestClassifier()
model.fit(x_train, y_train)

#------------------------------------------------prediction our training data
y_train_p = model.predict(x_train)
accur = accuracy_score(y_train, y_train_p)
print(accur,"this is accurancy of training data")

#------------------------------------------------prediction our test data
y_test_p = model.predict(x_test)
print(y_test_p, "this is our predivtion", y_test, "this is true values")
accur = accuracy_score(y_test, y_test_p)
print(accur, "this is prediction of test data")



