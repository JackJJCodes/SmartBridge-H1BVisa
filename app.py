from flask import Flask, request, url_for, redirect, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
PATH = 'modelRF.pkl'
model = pickle.load(open(PATH, 'rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def pred():
    inputFeatures = [float(x) for x in request.form.values()]
    featuresValue = [np.array(inputFeatures)]
    
    featureName = ['FULL_TIME_POSITION', 'PREVAILING_WAGE', 'YEAR','SOC_N']
    
    df = pd.DataFrame(featuresValue, columns=featureName)
    output = model.predict(df)
    print(output)
    
    return render_template('result.html', prediction_text = output)
        
if __name__ == '__main__':
    app.run(debug=True)
