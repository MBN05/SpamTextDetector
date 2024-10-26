from flask import Flask, render_template, request
import pandas as pd
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__, template_folder="templates")

# Load and prepare the data
data = pd.read_csv('spam.csv')
data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocess text data
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[\d+]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stop words and lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing to the messages
data['Message'] = data['Message'].apply(preprocess_text)

# Split the data
mess = data['Message']
cat = data['Category']
mess_train, mess_test, cat_train, cat_test = train_test_split(mess, cat, test_size=0.2)

# Feature extraction
cv = CountVectorizer()
features = cv.fit_transform(mess_train)

# Create and train the model
model = MultinomialNB()
model.fit(features, cat_train)

# Prediction function
def predict(message):
    input_message = cv.transform([preprocess_text(message)]).toarray()
    result = model.predict(input_message)
    return result[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    message = ""

    if request.method == 'POST':
        message = request.form['message']
        print(f"User Input: {message}")  # Log user input in the command line
        prediction = predict(message)

    return render_template('index.html', prediction=prediction, message=message)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
