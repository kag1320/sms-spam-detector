import nltk
import streamlit as st
import pickle
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def transform_text(text):
    lower_case_text = text.lower()
    tokenized_text = nltk.word_tokenize(lower_case_text)

    text_without_sp_char = []
    for text in tokenized_text:
        if text.isalnum():
            text_without_sp_char.append(text)

    text_without_stopwords_punctuation = []
    for text in text_without_sp_char:
        if text not in stopwords.words('english') and text not in string.punctuation:
            text_without_stopwords_punctuation.append(text)

    text_after_stemming = []
    for text in text_without_stopwords_punctuation:
        text_after_stemming.append(stemmer.stem(text))

    return " ".join(text_after_stemming)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Detector")
 
input_sms = st.text_area("Enter the message")
if st.button('predict'):
    # Preprocessing

    transformed_sms = transform_text(input_sms)

    # Vectorization
    vector_input = tfidf.transform([transformed_sms])

    # Prediction
    result = model.predict(vector_input)[0]

    # Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
