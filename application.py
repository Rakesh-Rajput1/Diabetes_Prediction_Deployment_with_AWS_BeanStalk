from flask import Flask,request,app,render_template
from flask import Flask
import pickle
import numpy as np
import pandas as pd


application = Flask(__name__)
app = application


scaler=pickle.load(open("Models/standardScalar.pkl","rb"))
model=pickle.load(open("Models/modelForPrediction.pkl","rb"))




# route for home page
@app.route('/')
def index():
    return render_template('index.html')



## route for single data prediction
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    result=""

    if request.method=='POST':

        Pregnancies= int(request.form.get('Pregnancies'))
        Glucose= float(request.form.get('Glucose'))
        BloodPressure= float(request.form.get('BloodPressure'))
        SkinThickness= float(request.form.get('SkinThickness'))
        Insulin= float(request.form.get('Insulin'))
        BMI= float(request.form.get('BMI'))
        DiabetesPedigreeFunction= float(request.form.get('DiabetesPedigreeFunction'))
        Age= int(request.form.get('Age'))

        
        new_data =scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        prediction=model.predict(new_data)

        if prediction[0]==1:
            result="The person is Diabetic" 
        else:
            result="The person is Non-Diabetic"

            # return render_template('index.html', prediction_text=result)
        return render_template('single_prediction.html',result=result)

    else:
        return render_template('home.html', prediction_text="Please enter the values")
    
if __name__ == "__main__":
    app.run(host="0.0.0.0")





