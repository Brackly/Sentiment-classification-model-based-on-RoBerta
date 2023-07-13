from model import get_sentiment
from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def hello():
    return {"status":"Healthy!"}

@app.route('/sentiment', methods=['POST'])
def sentiment():
    reqdata = request.get_json()
    text = reqdata['text']
    sentiment = get_sentiment(text)
    return {"response":sentiment}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)