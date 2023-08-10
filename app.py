from model import get_sentiment,get_comparison
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello():
    return {"status":"Healthy!"}

@app.route('/comparison', methods=['POST'])
def comparisons():
    reqdata = request.get_json()
    text1 = reqdata['text1']
    text2 = reqdata['text2']
    sentiment = get_sentiment(text1,text2)
    return sentiment

@app.route('/sentiment', methods=['POST'])
def sentiments():
    reqdata = request.get_json()
    text = reqdata['text']
    result = get_comparison(text)
    return result

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
