import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

import nltk
nltk.download()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import numpy as np
import _pickle as cPickle
import re
import flask

app = flask.Flask(__name__)
app.config["DEBUG"] = True

lemmatizer = WordNetLemmatizer()
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
stop_words = set(stopwords.words("english"))
maxLength=88
selected_categories = ['pl_usa', 'to_earn', 'to_acq', 'pl_uk', 'pl_japan', 'pl_canada', 'to_money-fx',
 'to_crude', 'to_grain', 'pl_west-germany', 'to_trade', 'to_interest',
 'pl_france', 'or_ec', 'pl_brazil', 'to_wheat', 'to_ship', 'pl_australia',
 'to_corn', 'pl_china']

with open(r"input_tokenizer.pickle", "rb") as input_file:
    input_tokenizer = cPickle.load(input_file)

def cleanUpSentence(r, stop_words = None):
    r = r.lower().replace("<br />", " ")
    r = re.sub(strip_special_chars, "", r.lower())
    if stop_words is not None:
        words = word_tokenize(r)
        filtered_sentence = []
        for w in words:
            w = lemmatizer.lemmatize(w)
            if w not in stop_words:
                filtered_sentence.append(w)
        return " ".join(filtered_sentence)
    else:
        return r


@app.route('/', methods=['GET'])
def classification():
	print("Please Enter some input")		
	input_sen = "hello worls"
	print("Cleaning up the sentence")		
	input = cleanUpSentence(input_sen, stop_words)
	textArray = np.array(pad_sequences(input_tokenizer.texts_to_sequences([input]), maxlen=maxLength))
	print("Loading Model from Disk")
	new_model = load_model("classification_model.h5")
	print("Predicting Categories")
	predicted = new_model.predict(textArray)[0]

	print("Given News Article belong to following Categories")
	for i, prob in enumerate(predicted):
		if prob > 0.2:
			print(selected_categories[i])
	return selected_categories[i]

	
if __name__ == '__main__':
    app.run(debug=True)