import pickle
from flask import Flask, render_template,request,app, jsonify
import numpy as np
import pandas as pd

app= Flask(__name__)
model= pickle.load(open("model1.pkl","rb"))

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/single_predict",methods=["POST"])

def single_predict():
    data=request.json['data']
    new= [list(data.values())]
    output=model.predict(new)[0]
    return jsonify(output)


@app.route("/predict",methods=["POST"])
def predict():
    data= [float(x) for x in request.form.values()]
    final= [np.array(data)]

    output=model.predict(final)[0]
    return render_template(
        'home.html', prediction_text=f"Airfoil pressure is {output}"
    )






if __name__=="__main__":
    app.run(debug=True)