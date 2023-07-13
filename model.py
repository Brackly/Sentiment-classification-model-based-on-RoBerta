from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax
import numpy as np


def fetch_reviews(text):
  # Read the reviews from the JSON file
  with open('reviews.json', 'r') as file:
      reviews = json.load(file)
  return reviews[text]


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
  res={}
  reviews=fetch_reviews(text)
  for index, review in reviews.items():
    encoded_input = tokenizer(review["text"], return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    sentiment=labels[ranking[0]]
    review["sentiment"]=sentiment
    review["ranking"]=ranking[0]
    res[index]=review
  return res


