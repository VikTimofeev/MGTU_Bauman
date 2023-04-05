
import pandas as pd
import numpy as np
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models
from keras.engine.sequential import Sequential
from keras.metrics import Precision, Recall
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
import pickle
from xgboost import XGBRegressor


from pyngrok import ngrok
from flask import Flask
import flask
import pickle
from flask import render_template
from joblib import load
import sklearn
from sklearn.linear_model import LinearRegression

auth = '2NlsksXPBfOF3gXvuyFwC1Biy8D_7gzxy8hLFyc3Piw6TnvJA'
# 2Nm3GGTjed1Dn0PTYWginfwR0wS_5w4mNWXb9TBYsD7B1iq3E API
# auth = 'Вставить сюда ключ для авторизации'
port_no = 5000

ngrok.set_auth_token(auth)
public_url = ngrok.connect(5000).public_url
app = flask.Flask(__name__, template_folder = 'templates')

# public_url = http://af91-46-0-224-0.ngrok.io

def f_name(n_model):
    if n_model == 1: return 'Linear Regression.pkl'
    if n_model == 2: return 'KNeighborsRegressor.pkl'
    if n_model == 3: return 'Lasso.pkl'
    if n_model == 4: return 'XGBoost.pkl'
    if n_model >= 5:return 'none'
    if n_model <= 0:return 'none'

def get_inputs():
    x1 = float(flask.request.form['input_parameter1'])
    x2 = float(flask.request.form['input_parameter2'])
    x3 = float(flask.request.form['input_parameter3'])
    x4 = float(flask.request.form['input_parameter4'])
    x5 = float(flask.request.form['input_parameter5'])
    x6 = float(flask.request.form['input_parameter6'])
    x7 = float(flask.request.form['input_parameter7'])
    x8 = float(flask.request.form['input_parameter8'])
    #x9 = float(flask.request.form['input_parameter9'])
    x10 = float(flask.request.form['input_parameter10'])
    x11 = float(flask.request.form['input_parameter11'])
    x12 = float(flask.request.form['input_parameter12'])
    return x1,x2,x3,x4,x5,x6,x7,x8,x10,x11,x12


@app.route('/', methods = ['POST', 'GET'])
@app.route('/index', methods = ['POST', 'GET'])
def main():
    if flask.request.method == 'GET':
        return render_template('main.html')
    
    if flask.request.method == 'POST':
        r_model = int(flask.request.form['input_parameter0'])
       # print('Выбрана модель:',r_model)
        file_name = ''
        file_name = f_name(r_model)
        if file_name=='none': 
            return render_template('main.html', result = 'Номер модели регрессии указан неверно')
        


        with open(file_name, 'rb') as f:
            loaded_model = pickle.load(f)
        with open('minmax_x.pkl', 'rb') as f:
            scaler_x = pickle.load(f)   
        with open('minmax_y.pkl', 'rb') as f:
            scaler_y = pickle.load(f)     

        X = get_inputs()
        X = scaler_x.transform(np.array(X).reshape(1,-1))
        y_pred = loaded_model.predict(X)
        y_pred = scaler_y.inverse_transform(y_pred)

        message = '1: Модуль упругости при растяжении = ' + str(int(y_pred[0,0]*10000)/10000) + ' ГПа; '  + '2:Прочность при растяжении = ' + str(int(y_pred[0,1]*10000)/10000) + ' МПа; '
        #return render_template('main.html', result = y_pred)
        return render_template('main.html', result = message)



print(f" Приложение по ссылке :  {public_url}")  
app.run(port = port_no)


        