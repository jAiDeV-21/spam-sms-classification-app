import streamlit as st
import pickle
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer


def transform_text(text):
    ps = PorterStemmer()
    text = text.lower()
    text = nltk.word_tokenize(text)

    text_transformed = []
    for word in text:
        if word.isalnum():
            text_transformed.append(word)

    text = text_transformed[:]
    text_transformed.clear()
    # passed copy of list because if directly assigned then after using
    # text_transformed.clear() 'text' list will become empty and nothing will be processed below

    for word in text:
        if word not in stopwords.words('english') and word not in string.punctuation:
            text_transformed.append(word)

    text = text_transformed[:]
    text_transformed.clear()

    # Stemming converts words like loving to love i.e, to a base word
    for word in text:
        text_transformed.append(ps.stem(word))

    return ' '.join(text_transformed)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
mnb_model = pickle.load(open('mnb-model.pkl', 'rb'))
st.title('Spam SMS Classifier')

input_sms = st.text_area('Enter the SMS')

if st.button('Predict'):
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_sms = tfidf.transform([transformed_sms]).toarray()

    # 3. Predict
    # result_voting = model.predict(vector_sms)[0]
    result_mnb = mnb_model.predict(vector_sms)[0]

    # 4. Display result

    if result_mnb == 1:
        st.header("Result by Multinomial Naive Bayes: Spam")
    else:
        st.header("Result by MultinomialNB: Not Spam")
