from flask import Flask,render_template,request

import FeatureExtraction
import joblib

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("home.html")

@app.route('/about')
def about():    
    return render_template("about.html")

@app.route('/getURL',methods=['GET','POST'])
def getURL():
    if request.method == 'POST':
        url = request.form['url']
        print(url)
        data = FeatureExtraction.getAttributes(url)
        print(data)
        svm_model = joblib.load(r'Phishing-Website-Detection-master\Phishing website detection using UI\model1.pkl')
        predicted_value = svm_model.predict(data)
        print(predicted_value)
        if predicted_value == 0:    
            value = "Legitimate"
            return render_template("home.html",error=value)
        else:
            value = "Phishing"
            return render_template("home.html",error=value)
if __name__ == "__main__":
    app.run(debug=True)