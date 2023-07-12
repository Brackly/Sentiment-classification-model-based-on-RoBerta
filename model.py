from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax
import numpy as np

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
  encoded_input = tokenizer(text, return_tensors='pt')
  output = model(**encoded_input)
  scores = output[0][0].detach().numpy()
  scores = softmax(scores)


  ranking = np.argsort(scores)
  ranking = ranking[::-1]
  res=labels[ranking[0]]
  return res