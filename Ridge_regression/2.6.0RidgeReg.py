import pandas as pd

data = pd.read_csv("C:/Users/kekli/Desktop/ML event/reklam.csv")
veri= data.copy()
#print(veri)

y=veri["Sales"]
X=veri.drop(columns="Sales",axis=1)
#print(veri.isnull().sum())

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression
import sklearn.metrics as mt

lr=LinearRegression()
lr.fit(X_train,y_train)
tahmin=lr.predict(X_test)

r2 = mt.r2_score(y_test,tahmin)
mse = mt.mean_squared_error(y_test,tahmin)
print("R2 Score:{} MSE:{}".format(r2,mse))

#Normal modeli kurduk.
#Şimdi ridge modeli kurdumada
from sklearn.linear_model import Ridge

ridge_model=Ridge(alpha=150)
ridge_model.fit(X_train,y_train)
tahmin2=ridge_model.predict(X_test)

r2_rid = mt.r2_score(y_test,tahmin2)
mse_rid = mt.mean_squared_error(y_test,tahmin2)
print("R2_rid Score:{} MSE_rid:{}".format(r2_rid,mse_rid))


import matplotlib.pyplot as plt
import numpy as np

katsayılar=[]
lambdalar=10**np.linspace(10,-2,100)*0.5

for i in lambdalar:
    ridmodel=Ridge(alpha=i)
    ridmodel.fit(X_train,y_train)
    katsayılar.append(ridmodel.coef_)

ax=plt.gca()
ax.plot(katsayılar,lambdalar)
ax.set_xscale("log")
plt.xlabel("Lambda")
plt.ylabel("Katsayılar")
plt.show()






