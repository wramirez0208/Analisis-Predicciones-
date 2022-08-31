# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 16:47:54 2022

@author: wilma
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

print("Current Working Directory " , os.getcwd())

os.chdir("C:/Users/wilma/Desktop/Maestria Estadistica Aplicada/8. Octavo Trimestre/Analisis Predictivo de Negocios")
print("Current Working Directory " , os.getcwd())

data = pd.DataFrame(pd.read_csv("USA_Housing.csv"))

data.head()

len(data)
data.dtypes
data.describe()

""""luego de conocer los datos, realizamos los graficos"""
data.Price.hist() 
plt.title('Distribución de Precios')


""""con el proposito de realizar la regresion de manera correcta, excluimos la variable direccion"""""

data.drop(['Address'],axis=1,inplace=True)
data.describe()
data.head()

""""Se grafica la variable dependiente precio de las casas en USA"""""

sns.distplot(data['Price'], color='b')
plt.title('Distribución de Precios Housing Usa')
""""Se puede observar que la variable precio de las casas en USA, posee una distribución normal"""""


sns.heatmap(data.corr(),annot=True)
plt.title('Correlación entre las variables de Housing Usa')

from sklearn import preprocessing
pre_process = preprocessing.StandardScaler()

"""se separan los datos para poder realizar las regresiones de lugar en entrenamiento y prueba"""
X = data[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population']]

# Putting response variable to y
y = data['Price']


X = pd.DataFrame(pre_process.fit_transform(X))

X.head()

y.head()


"""se realiza con un 30% para entrenamiento y el resto para prueba"""
from sklearn.model_selection import train_test_split

X_train,  X_test, y_train, y_test = train_test_split(X, 
                                                     y, 
                                                     train_size=0.3, 
                                                     random_state=0)


X_train.shape, y_train.shape, X_test.shape, y_test.shape

"""utilizando regresion lineal sin regularizacion, y calculando los errores para los datos de entrenamiento y de prueba"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import PolynomialFeatures

lin_mod = LinearRegression()
lin_mod.fit(X_train, y_train)
print(mse(y_test, lin_mod.predict(X_test)))
print(mse(y_train, lin_mod.predict(X_train)))

print(lin_mod .intercept_)

coeff_df = pd.DataFrame(lin_mod,X.columns,columns=['Coefficient'])
coeff_df


coef_lin = pd.DataFrame(lin_mod .coef_,X_test.columns,columns=['Coefficient'])
coef_lin

"""Observando las variables anteriores 
'Avg. Area Income','Avg. Area House Age',
'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms',
'Area Population']]
"""

y_pred = lin_mod.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)


"""ERORRES"""
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)


from math import sqrt

rms = sqrt(mse)
rms

#coeficientes
coeff_df = pd.DataFrame(lin_mod.coef_,X.columns,columns=['Coefficient'])
coeff_df

"""ESTE MODELO POSEE UN ERROR MUY ELEVADO, SE DEBE DE MEJORAR EL MODELO"""

 
"utilizando Statsmodels para poder tener un summary del modelo"

import statsmodels.api as sm
X_train_sm = X_train
X_train_sm = sm.add_constant(X_train_sm)
lm_1 = sm.OLS(y_train,X_train_sm).fit()


"""coeficientes"""
lm_1.params

print(lm_1.summary())

"""con el fin de mejorar el modelo, vamos a eliminar la variable 3, area de los cuartos, Avg. Area Number of Rooms"""


X.head()

X.drop([3],axis=1, inplace=True)

X.head()

"""continuamos a probar el modelo"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 ,test_size = 0.3, random_state=2)


print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)


from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train, y_train)


print(lm.intercept_)


"coeficientes"
coeff_df = pd.DataFrame(lm.coef_,X_test.columns,columns=['Coefficient'])
coeff_df

y_pred = lm.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)

from math import sqrt

rms = sqrt(mse)
rms

"summary del modelo"

import statsmodels.api as sm
X_train_sm2 = X_train
X_train_sm2 = sm.add_constant(X_train_sm2)
lm_2 = sm.OLS(y_train,X_train_sm2).fit()


"""coeficientes"""
lm_2.params

print(lm_2.summary())







