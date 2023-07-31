from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax
import numpy as np
import json

def fetch_reviews(text):
  # Read the reviews from the JSON file
  with open('amazon.json', 'r') as file:
      reviews = json.load(file)
  return [review for review in reviews if review["asin"] == text]

def fetch_preds():
    with open('preds.json', 'r') as file:
      preds = json.load(file)
    return preds

def get_sentiment(text):
  task='sentiment'
  MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

  tokenizer = AutoTokenizer.from_pretrained(MODEL)
  model = AutoModelForSequenceClassification.from_pretrained(MODEL)

  labels={
      0:'negative',
      1:'neutral',
      2:'positive'
  }

  # model.save_pretrained(MODEL)
  res=[]
  reviews=fetch_reviews(text)
  if reviews == None:
      return {"response":"Comments for the querry do not appear in the database"}
  else:
      for review in reviews:
          if len(review["text"]) >2000:
             review["text"] = review["text"][0:500]
          encoded_input = tokenizer(review["text"], return_tensors='pt')
          output = model(**encoded_input)
          scores = output[0][0].detach().numpy()
          scores = softmax(scores)
          ranking = np.argsort(scores)
          ranking = ranking[::-1]
          sentiment=labels[ranking[0]]
          review["sentiment"]=sentiment
          review["ranking"]=str(ranking[0])
          res.append(review)
  return res
