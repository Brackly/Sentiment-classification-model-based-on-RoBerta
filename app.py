from model import get_sentiment,fetch_preds
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello():
    return {"status":"Healthy!"}

@app.route('/sentiment', methods=['POST'])
def sentiments():
    reqdata = request.get_json()
    text = reqdata['text']
    sentiment = get_sentiment(text)
    return {"response":sentiment}

@app.route('/predictions', methods=['POST'])
def sentiment():
    return {"response":fetch_preds()}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
