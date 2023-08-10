from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax
import numpy as np
import json
#------------------------------------------------------------------------------------------

def classify(reviews):
  task='sentiment'
  MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

  tokenizer = AutoTokenizer.from_pretrained(MODEL)
  model = AutoModelForSequenceClassification.from_pretrained(MODEL)

  labels={
    0:'negative',
    1:'neutral',
    2:'positive'
  }
  rev=[]
  for review in reviews:
        if len(review["text"])>500:
            review["text"] = review["text"][0:500]
        # print()
        encoded_input = tokenizer(review["text"], return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        sentiment=labels[ranking[0]]
        review["sentiment"]=sentiment
        review["ranking"]=str(ranking[0])
        rev.append(review)
  return rev

def fetch_reviews(text):
  # Read the reviews from the JSON file
  with open('amazon.json', 'r') as file:
      reviews = json.load(file)
  return [review for review in reviews if text.lower() in review["asin"].lower() ]

#---------------------------------------------------------------------------------------------

def get_sentiment(text):
    reviews=fetch_reviews(text)
    res={}
    res[text] = classify(reviews)
    return res

def get_comparison(text1,text2):
  # model.save_pretrained(MODEL)
  reviews1=fetch_reviews(text1)
  reviews2=fetch_reviews(text2)
  if reviews1 == None and reviews2==None:
      return {"response":"Comments for the querry do not appear in the database"}
  else:
      res={}
      res[text1] = classify(reviews1)
      res[text2] = classify(reviews2)
  return res

print(get_comparison("ipad","iphone"))