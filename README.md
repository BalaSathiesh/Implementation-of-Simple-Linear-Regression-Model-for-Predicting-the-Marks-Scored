# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: 
RegisterNumber:  
*/

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
![ML EX 02_page-0001](https://github.com/BalaSathiesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128462891/d4920317-21e9-4add-b1b0-12ea0cb6fa1a)
![ML EX 02_page-0002](https://github.com/BalaSathiesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128462891/0e145448-aaeb-44d9-8a2a-c15731627cd0)
![ML EX 02_page-0003](https://github.com/BalaSathiesh/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128462891/527aef22-fc59-4746-b57f-8e8baa72552d)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
