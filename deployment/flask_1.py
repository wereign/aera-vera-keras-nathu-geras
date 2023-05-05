import joblib
import numpy as np
from tensorflow.keras.models import load_model

from flask import Flask, request, jsonify


flower_model = load_model("C:\\Users\\viren\\Desktop\\TensorFlow Course\\deployment\\iris_final.h5")
flower_scaler = joblib.load('C:\\Users\\viren\\Desktop\\TensorFlow Course\\deployment\\iris_scaler.pkl')

def return_prediction(model,scaler,sample_json):
    s_len = sample_json['sepal_length']
    s_wid = sample_json['sepal_width']
    p_len = sample_json['petal_length']
    p_wid = sample_json['petal_width']

    flower = [[s_len,s_wid,p_len,p_wid]]

    flower = scaler.transform(flower)
    
    classes_array = np.array(['setosa', 'versicolor', 'virginica'])
    class_ind =  np.argmax(model.predict(flower))

    return classes_array[class_ind]

app = Flask(__name__)

@app.route("/")
def index():
    return "<h1>Flask App!</h1>"

@app.route("/api/flower",methods=["POST"])
def flower_prediction():

    content = request.json
    results = return_prediction(flower_model,flower_scaler,content)

    return jsonify(results)


if __name__ ==  "__main__":
    app.run()