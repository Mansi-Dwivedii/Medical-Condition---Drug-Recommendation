from codecs import ignore_errors
from distutils.core import setup
from doctest import REPORT_ONLY_FIRST_FAILURE
from fileinput import filename
import streamlit as st
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import joblib
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import WordNetError
from PIL import Image
import sys

Model_PATH = "PACmodel.pkl"

Tokenizer_PATH = "tfidfvectorizer.pkl"

DATA_PATH ="drugsCom.csv"

import nltk

nltk.download('stopwords')

def is_good_word(word):
    word = word.strip()
    if len(word) <= 2:
        return 0
    if word in stop_words:
        return 0
    try:
        if len(wn.lemmas(str(word), lang='en')) == 0:     # no longer the first access of wn
            return 0
    except WordNetError as e:
        print("WordNetError on concept {}".format(word))
    except AttributeError as e:
        print("Attribute error on concept {}: {}".format(word, e.message))
    except:
        print("Unexpected error on concept {}: {}".format(word, sys.exc_info()[0]))
    else:
        return 1
    return 1

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

st.set_page_config(page_title="Medical Condition - Drug Recommendation", page_icon=":dna:", layout="centered")

vectorizer = joblib.load(Tokenizer_PATH) #Loading Vectorizer
model = joblib.load(Model_PATH) #Loading Model

def review_words(raw_review):
	# Deleting HTML
	review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
	# Make a Space
	letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
	# Lower letters
	words = letters_only.lower().split()
	# stopwords
	meaningful_words = [w for w in words if not w in stop_words]
	# Lemmitization
	lemmitize_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
	# Space join words
	return(' '.join(lemmitize_words))

def TopDrugs(condition, df):
	df_top = df[(df['rating']>=9)&(df['usefulCount']>=100)].sort_values(by = ['rating', 'usefulCount'], ascending = [False, False])
	drug_lst = df_top[df_top['condition']==condition]['drugName'].head(3)
	return drug_lst

def main():
	review = st.text_input("Enter Your Medical Condition", max_chars=1000)
	submit = st.button("Submit")
	X = pd.DataFrame({'Review':[review]})
	if submit:
		X['cleaned_review'] = X["Review"].apply(review_words)
		cleaned = X['cleaned_review']
		st.subheader("User Medical Review")
		st.write("Review: ",X["Review"][0])
		tfidf_vect = vectorizer.transform(cleaned)
		prediction = model.predict(tfidf_vect)
		predicted_condition = prediction[0]
		df = pd.read_csv(DATA_PATH)
		top_drugs = TopDrugs(predicted_condition,df)
		
		# ---- Visualization ----
		st.subheader("Predicted Medical Condition")
		st.write("Condition: ",predicted_condition)
		st.subheader("Top 3 Recommended Drugs based on Condition Review")
		st.write("Drug 1: ",top_drugs.iloc[0])
		st.write("Drug 2: ",top_drugs.iloc[1])
		st.write("Drug 3: ",top_drugs.iloc[2])
	else:
		st.write("Enter review & press submit")
	#return main()

# --- USER AUTHENTICATION ---
st.title(":blue[Medical Condition - Drug Recommendation] :dna:")
menu = ["Login", "Logout"]

choice = st.sidebar.selectbox("Login/Logout as Admin",menu)
if choice == "Login":
	st.sidebar.subheader("Login as Admin")
	username = st.sidebar.text_input("User Name")
	password = st.sidebar.text_input("Password",type="password")
	if st.sidebar.button("Login"):
		if password == "Admin@23":
			st.success("Logged in as {}".format(username))
			main()
		elif password != "Admin@23":
			st.warning("Incorrect Username/Password")
	else:
		image = Image.open('drugimg.jpg')
		st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
		#st.write("Error has occured !")
	try:
		st.warning("Duplicated values, Please click on Submit")
	except:
		if __name__ == '__main__':
			main()
elif choice == "Logout":
	image = Image.open('drugimg.jpg')
	st.image(image, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
	st.success("Logged out Successfully")
