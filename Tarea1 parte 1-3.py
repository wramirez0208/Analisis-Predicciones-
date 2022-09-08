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


"""logit""" 

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)

from __future__ import print_function
from datetime import datetime
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
import keras.backend as K
from keras.wrappers.scikit_learn import KerasClassifier


#utility functions
def plot_decision_boundary(func, X, y, figsize=(9, 6)):
    amin, bmin = X.min(axis=0) - 0.1
    amax, bmax = X.max(axis=0) + 0.1
    hticks = np.linspace(amin, amax, 101)
    vticks = np.linspace(bmin, bmax, 101)
    
    aa, bb = np.meshgrid(hticks, vticks)
    ab = np.c_[aa.ravel(), bb.ravel()]
    c = func(ab)
    cc = c.reshape(aa.shape)

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    
    fig, ax = plt.subplots(figsize=figsize)
    contour = plt.contourf(aa, bb, cc, cmap=cm, alpha=0.8)
    
    ax_c = fig.colorbar(contour)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, 0.25, 0.5, 0.75, 1])
    
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
    plt.xlim(amin, amax)
    plt.ylim(bmin, bmax)

def plot_multiclass_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))
    cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
#linea 71 con error que debe ser corregido
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()], verbose=0)
    Z = Z.reshape(xx.shape)
    fig = plt.figure(figsize=(8, 8))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    
def plot_data(X, y, figsize=None):
    if not figsize:
        figsize = (8, 6)
    plt.figure(figsize=figsize)
    plt.plot(X[y==0, 0], X[y==0, 1], 'or', alpha=0.5, label=0)
    plt.plot(X[y==1, 0], X[y==1, 1], 'ob', alpha=0.5, label=1)
    plt.xlim((min(X[:, 0])-0.1, max(X[:, 0])+0.1))
    plt.ylim((min(X[:, 1])-0.1, max(X[:, 1])+0.1))
    plt.legend()

def plot_loss_accuracy(history):
    historydf = pd.DataFrame(history.history, index=history.epoch)
    plt.figure(figsize=(8, 6))
    historydf.plot(ylim=(0, max(1, historydf.values.max())))
    loss = history.history['loss'][-1]
    acc = history.history['acc'][-1]
    plt.title('Loss: %.3f, Accuracy: %.3f' % (loss, acc))

def plot_loss(history):
    historydf = pd.DataFrame(history.history, index=history.epoch)
    plt.figure(figsize=(8, 6))
    historydf.plot(ylim=(0, historydf.values.max()))
    plt.title('Loss: %.3f' % history.history['loss'][-1])
    
def plot_confusion_matrix(model, X, y):
    y_pred = (model.predict(X) > 0.5).astype("int32")
    plt.figure(figsize=(8, 6))
    sns.heatmap(pd.DataFrame(confusion_matrix(y, y_pred)), annot=True, fmt='d', cmap='YlGnBu', alpha=0.8, vmin=0)

def plot_compare_histories(history_list, name_list, plot_accuracy=True):
    dflist = []
    for history in history_list:
        h = {key: val for key, val in history.history.items() if not key.startswith('val_')}
        dflist.append(pd.DataFrame(h, index=history.epoch))

    historydf = pd.concat(dflist, axis=1)

    metrics = dflist[0].columns
    idx = pd.MultiIndex.from_product([name_list, metrics], names=['model', 'metric'])
    historydf.columns = idx
    
    plt.figure(figsize=(6, 8))

    ax = plt.subplot(211)
    historydf.xs('loss', axis=1, level='metric').plot(ylim=(0,1), ax=ax)
    plt.title("Loss")
    
    if plot_accuracy:
        ax = plt.subplot(212)
        historydf.xs('acc', axis=1, level='metric').plot(ylim=(0,1), ax=ax)
        plt.title("Accuracy")
        plt.xlabel("Epochs")

    plt.tight_layout()
    
def make_sine_wave():
    c = 3
    num = 2400
    step = num/(c*4)
    np.random.seed(0)
    x0 = np.linspace(-c*np.pi, c*np.pi, num)
    x1 = np.sin(x0)
    noise = np.random.normal(0, 0.1, num) + 0.1
    noise = np.sign(x1) * np.abs(noise)
    x1  = x1 + noise
    x0 = x0 + (np.asarray(range(num)) / step) * 0.3
    X = np.column_stack((x0, x1))
    y = np.asarray([int((i/step)%2==1) for i in range(len(x0))])
    return X, y

def make_multiclass(N=500, D=2, K=3):
    """
    N: number of points per class
    D: dimensionality
    K: number of classes
    """
    np.random.seed(0)
    X = np.zeros((N*K, D))
    y = np.zeros(N*K)
    for j in range(K):
        ix = range(N*j, N*(j+1))
        # radius
        r = np.linspace(0.0,1,N)
        # theta
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu, alpha=0.8)
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    return X, y




#DATOS
print("Current Working Directory " , os.getcwd())

os.chdir("C:/Users/wilma/Desktop/Maestria Estadistica Aplicada/8. Octavo Trimestre/Analisis Predictivo de Negocios")
print("Current Working Directory " , os.getcwd())

data = pd.DataFrame(pd.read_csv("USA_Housing.csv"))

from sklearn.model_selection import train_test_split



X_train,  X_test, y_train, y_test = train_test_split(X, 
                                                     y, 
                                                     train_size=0.5, 
                                                     random_state=0)


X_train.shape, y_train.shape, X_test.shape, y_test.shape


#Modelo 1- Logistic Regresion
log_reg = LogisticRegression(C=1, solver='lbfgs', multi_class='multinomial')
log_reg.fit(X_train, y_train)

train_X2 = sm.add_constant(train_X, prepend=True)
model2 = sm.OLS(endog=train_Y, exog=train_X2,)
model2= model2.fit()
print(model2.summary())

#predicciones
test_predictions = log_reg.predict(test_X)

#Accuraccy
accuracy_score(test_predictions, test_Y)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_Y, test_predictions)
cm


"""Partimos a realizar una regresion logistica"""



%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)

from __future__ import print_function
from datetime import datetime
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
import keras.backend as K
from keras.wrappers.scikit_learn import KerasClassifier


X.head()

y.head()



import statsmodels.api as sm 
import pandas as pd 


import statsmodels.api as sm
X_train_sm3 = X_train
X_train_sm3 = sm.add_constant(X_train_sm2)
lm_2 = sm.OLS(y_train,X_train_sm2).fit()
log_reg = sm.Logit(y_train, X_train_sm3).fit() 


lr = LogisticRegression()
lr.fit(X, y)
print('LR coefficients:', lr.coef_)
print('LR intercept:', lr.intercept_)

plot_data(X, y)

limits = np.array([-2, 2])
boundary = -(lr.coef_[0][0] * limits + lr.intercept_[0]) / lr.coef_[0][1]
plt.plot(limits, boundary, "g-", linewidth=2)







"utilizando Statsmodels para poder tener un summary del modelo"

import statsmodels.api as sm
X_train_sm = X_train
X_train_sm = sm.add_constant(X_train_sm)
lm_1 = sm.OLS(y_train,X_train_sm).fit()


"""coeficientes"""
lm_1.params

print(lm_1.summary())






""""RNN"""

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




from sklearn.naive_bayes import GaussianNB
#Modelo
rnn_reg = GaussianNB()
rnn_reg.fit(X_train, y_train)

#predicciones
rnn_reg_predictions = rnn_reg.predict(X_test)

#Accuraccy
accuracy_score(rnn_reg_predictions, y_test)
cm2 = confusion_matrix(y_test, rnn_reg_predictions)
cm2




