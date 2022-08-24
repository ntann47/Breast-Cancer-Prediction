
from flask import Flask, redirect, url_for, render_template, request
import pickle
import cv2
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
app = Flask(__name__)

# path = os.path.join(cwd, "DeepLearning/static/images")
# app.config['IMAGES_FOLDER'] = path

@app.route('/', methods=['GET'])

def hello_world():
    return render_template('index.html')
@app.route('/', methods=['POST'])
def predict():
    fields = [k for k in request.form]
    values = [request.form[k] for k in request.form]
    data = dict(zip(fields,values))
    try:   
   
        list_features = []
        for key, value in data.items():
            temp = value
            list_features.append(temp)
        list_features = [float(x) for x in list_features]
        print(list_features)
        print(type(list_features))


        data1 = pd.read_csv('breast-cancer.csv')
        data1.replace({"diagnosis":{'M':1,'B':0}},inplace=True)
        data1 = data1.astype(float)
        X=data1.iloc[:500,2:32]
        Y=data1.iloc[:500,1]

        for col in X.columns:
            X[col][np.isinf(X[col])]=X[col].mean()

        selected_features = ['concavity_worst', 'concavity_mean', 'area_mean', 'radius_mean',
        'area_worst', 'perimeter_mean', 'radius_worst', 'concave points_mean',
        'perimeter_worst', 'concave points_worst']
        X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
        X_train = X_train[selected_features]
            #Feature Scaling
        scaler=StandardScaler()
        X_train=scaler.fit_transform(X_train.values)
        features = scaler.transform(np.array(list_features).reshape(1,-1))
        print(features)

        model = pickle.load(open("breastcancer_prediction_model.pkl", "rb"))
    # use model to predict
        prediction = model.predict(features)
        if prediction == 0:
            prediction = 'Benign - Lành tính'
        else :
            prediction = 'Malignant - Ác tính'
    except :
        prediction = ' '
    print(prediction)
    return render_template('index.html', predict = prediction)


    



if __name__ == "__main__":
    app.run(debug=True)

# print(app)