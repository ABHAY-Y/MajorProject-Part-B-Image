from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
import easyocr
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from bs4 import BeautifulSoup # Text Cleaning
import re, string # Regular Expressions, String
from nltk.corpus import stopwords # stopwords
from nltk.stem.porter import PorterStemmer # for word stemming
from nltk.stem import WordNetLemmatizer # for word lemmatization
import unicodedata
import html
import nltk
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import json
import numpy
app = Flask(__name__)
CORS(app)
nltk.download('stopwords')
app = Flask(__name__)
CORS(app)
modelfordvdetectionforweb = tf.keras.models.load_model("D:\TextProj\dvdtect_model.h5")
stop = set(stopwords.words('english'))

# update stopwords to have punctuation too
stop.update(list(string.punctuation))

def clean_text(text):

    # Remove unwanted html characters
    re1 = re.compile(r'  +')
    x1 = text.lower().replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
    'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
    '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
    ' @-@ ', '-').replace('\\', ' \\ ')
    text = re1.sub(' ', html.unescape(x1))

    # remove non-ascii characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

#     # strip html
#     soup = BeautifulSoup(text, 'html.parser')
#     text = soup.get_text()

    # remove between square brackets
    text = re.sub('\[[^]]*\]', '', text)

    # remove URLs
    text = re.sub(r'http\S+', '', text)

    # remove twitter tags
    text = text.replace("@", "")

    # remove hashtags
    text = text.replace("#", "")

    # remove all non-alphabetic characters
    text = re.sub(r'[^a-zA-Z ]', '', text)

    # remove stopwords from text
    final_text = []
    for word in text.split():
        if word.strip().lower() not in stop:
            final_text.append(word.strip().lower())

    text = " ".join(final_text)

    # lemmatize words
    lemmatizer = WordNetLemmatizer()
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    text = " ".join([lemmatizer.lemmatize(word, pos = 'v') for word in text.split()])

    # replace all numbers with "num"
    text = re.sub("\d", "num", text)

    return text.lower()

@app.route('/imageinterpage',methods=['POST'])
def connect():
    val = request.json
    proval = val.split(";base64,")[1]
    actinpt = Image.open(BytesIO(base64.b64decode(proval)))
    eyov = easyocr.Reader(['en'],gpu=False)
    inpt = eyov.readtext(actinpt)
    otpt = ' '.join([detect[1] for detect in inpt])
    print(otpt)
    vocab_size = 10000
    test_median_data = int(18.0)
    cleaned_sentence = clean_text(otpt)
    tokenizer = Tokenizer(num_words = vocab_size, oov_token = 'UNK')
    tokenizer.fit_on_texts(cleaned_sentence)
    X_input = tokenizer.texts_to_sequences([cleaned_sentence])
    X_input = pad_sequences(X_input, maxlen=test_median_data, truncating='post', padding='post')
    # Make the prediction
    prediction = modelfordvdetectionforweb.predict(X_input)
    # Display the prediction for each label
    summationofvalus = prediction[0][0]+prediction[0][1]+prediction[0][2]+prediction[0][2]+prediction[0][3]+prediction[0][4]
    if summationofvalus <1.0:
        answer = "No Domestic Violence detected"
        pred = {"predicted":answer}
        return jsonify(pred)
    else:
        prediction_float = prediction[0].tolist()
        predicnjn = {
        'severe_toxicity': prediction_float[0],
        'obscene': prediction_float[1],
        'threat': prediction_float[2],
        'insult': prediction_float[3],
        'identity_attack': prediction_float[4]
        }
        ans = {"predicted":predicnjn}
        response_json = json.dumps(ans)
        return response_json
if __name__=='__main__':
    app.run(host='localhost',port=4000)    