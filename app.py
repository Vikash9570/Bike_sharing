import numpy as np
import pandas as pd
from flask import Flask, request, render_template

from src.pipeline.perdiction_pipeline import CustomData, PredictionPipeline

app = Flask(__name__)

# Load pre-trained model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction_data():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data=CustomData(
            Hour=float(request.form.get('Hour')),
            Temperature = float(request.form.get('Temperature')),
            Humidity = float(request.form.get('Humidity')),
            Windspeed = float(request.form.get('Windspeed')),
            Visibility = float(request.form.get('Visibility')),
            Seasons = float(request.form.get('Seasons')),
            Holiday = request.form.get('Holiday'),
            FunctioningDay= request.form.get('FunctioningDay'),
            Day = request.form.get('Day'),
            Month = request.form.get('Month'),
            Year= request.form.get('Year')
        )
        new_data= data.get_data_as_dataframe()
        predict_pipe=PredictionPipeline()
        pred=predict_pipe.predict(new_data)

        return render_template('results.html',final_result=pred)

if __name__=="__main__":
    app.run(host='127.0.0.1',debug=True)

