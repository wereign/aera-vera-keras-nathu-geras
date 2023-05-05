import joblib
import numpy as np
from tensorflow.keras.models import load_model

from flask import Flask, render_template,session,url_for,redirect
from flask_wtf import FlaskForm
from wtforms import StringField,SubmitField


flower_model = load_model(".\iris_final.h5")
flower_scaler = joblib.load('.\iris_scaler.pkl')

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
app.config['SECRET_KEY'] = 'pikachu'


class FlowerForm(FlaskForm):
    
    sep_len = StringField("Sepal length")
    sep_wid = StringField("Sepal Width")
    pet_len = StringField("Petal length")
    pet_wid = StringField("Petal Width")

    submit = SubmitField("Analyze")

@app.route("/",methods=['GET','POST'])
def index():
    form = FlowerForm()

    if form.validate_on_submit():

        session['sep_len'] = form.sep_len.data
        session['sep_wid'] = form.sep_wid.data
        session['pet_len'] = form.pet_len.data
        session['pet_wid'] = form.pet_wid.data

        return redirect(url_for("prediction"))
    return render_template("home.html",form=form)

@app.route("/prediction")
def prediction():
    content = {}

    content['sepal_length'] = float(session['sep_len'])
    content['sepal_width'] = float(session['sep_wid'])
    content['petal_length'] = float(session['pet_len'])
    content['petal_width'] = float(session['pet_wid'])

    results = return_prediction(flower_model,flower_scaler,content)

    return render_template('prediction.html',results=results)

if __name__ ==  "__main__":
    app.run()