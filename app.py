from flask import Flask,render_template,request
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
import re
import pickle
app = Flask(__name__)

classifier = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('transform.pkl','rb'))

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/submit',methods=['POST'])
def submit():
    if request.method == "POST":
        tweet = request.form['tweet']
        tweet =  " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ",tweet.lower()).split())
        tweet = nltk.word_tokenize(tweet)
        stemmer = PorterStemmer()
        stem = [stemmer.stem(word) for word in tweet]
        words = [word for word in stem if word not in stopwords.words('english')]
        tweet = " ".join(words)
        vect = cv.transform([tweet])
        my_prediction = classifier.predict(vect)
        return render_template("welcome.html", score = my_prediction)

if __name__ == '__main__':
    app.run(debug=True)
