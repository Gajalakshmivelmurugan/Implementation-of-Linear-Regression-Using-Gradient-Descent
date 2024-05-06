# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.
 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: gajalakshmi V
RegisterNumber:  212223040047
*/
/*
Program to implement the linear regression using gradient descent.
Developed by: sbiraj e
RegisterNumber:
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        
        #calculate predictions
        predictions = (X).dot(theta).reshape(-1,1)
        
        #calculate errors
        errors=(predictions - y ).reshape(-1,1)
        
        #update theta using gradiant descent
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
                                        
data=pd.read_csv("C:/classes/ML/50_Startups.csv")
data.head()

#assuming the lost column is your target variable 'y' 

X = (data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler = StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X)
print(X1_Scaled)

#learn modwl paramerers

theta=linear_regression(X1_Scaled,Y1_Scaled)

#predict target value for a new data
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")

```

## Output:

![316402749-bb90151f-89d9-47b2-831d-b0b97453b3bb](https://github.com/Gajalakshmivelmurugan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144871940/be67c480-481a-490f-bb3e-a52ee21e100d)
![316403401-a8efedff-9882-428e-b08b-436e9ca815fb](https://github.com/Gajalakshmivelmurugan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144871940/c4528c06-b9cb-42d1-a0c6-800a44d2a7be)
![316403437-aaa528c5-65ee-4bfc-85a3-d273fad084ca](https://github.com/Gajalakshmivelmurugan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144871940/e1d73bf3-767c-49f0-8036-9126b1e3c448)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
