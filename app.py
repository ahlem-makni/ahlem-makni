from flask import Flask, render_template, request
import pickle
import joblib
import numpy as np

model = pickle.load(open('arbreDeDecision.pkl', 'arbreDeDecision'))
randomForest = pickle.load(open('randomForest.pkl', 'randomForest'))
NaiveBayes = pickle.load(open('NaiveBayes.pkl', 'NaiveBayes'))
LogisticRegression = pickle.load(open('LogisticRegression.pkl', 'LogisticRegression'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['Heart_Disease']
    data2 = request.form['BMI']
    data3 = request.form['Smoking']
    data4 = request.form['Alcohol_Drinking']
    data5 = request.form['Stroke']
    data6 = request.form['Physical_Health']
    data7 = request.form['Mental_Health']
    data8 = request.form['Diff_Walking']
    data9 = request.form['sex']
    data10 = request.form['Age_Category']
    data11= request.form['Race']
    data12= request.form['Diabetic']
    data13= request.form['Physical_Activity']
    data14= request.form['Gen_Health']
    data15= request.form['Sleep_Time']
    data16= request.form['Asthma']
    data17= request.form['Kidney_Disease']


    arr = np.array([[data1, data2, data3, data4, data5, data6, data7,data8,data9,data10,data11,data12,data13,data14,data15,data16,data17]])
    pred = model.predict(arr)
  
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)















