# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries. 
2. Set variables for assigning dataset values. 
3. Import linear regression from sklearn. 
4. Assign the points for representing in the graph. 
5. Predict the regression for marks by using the representation of the graph. 
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
import pandas as pd
df=pd.read_csv('/content/Untitled spreadsheet - Sheet1.csv')
df
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/Untitled spreadsheet - Sheet1.csv')
plt.xlabel('X')
plt.xlabel('Y')
plt.scatter(df['x'],df['y'])
plt.xlabel('X')
plt.xlabel('Y')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
x
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
x_train
y_train
lr.predict(x_test.iloc[0].values.reshape(1,1))
plt.scatter(df['x'],df['y'])
plt.xlabel('x')
plt.xlabel('y')
plt.plot(x_train,lr.predict(x_train),color='red')
```
## Output:
## Head
![image](https://github.com/BalaSathiesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128462891/798f4d67-151c-4f50-a205-f654229f6851)

## Graph of Plotted Data
![image](https://github.com/BalaSathiesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128462891/e96258d6-db75-4b4d-a243-f7b14a461349)

## Trained Data
![image](https://github.com/BalaSathiesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128462891/37a5c732-ead6-4d4c-850d-59c5201a473d)

## Line Of Regression
![image](https://github.com/BalaSathiesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128462891/c25e36bd-cc45-4477-bdcc-743d69dbaac0)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
