import pickle
import pandas as pd

import requests
import streamlit as st

# Preprocess input
from utils import text_preprocessing

@st.cache(allow_output_mutation=True)
def load_session():
    return requests.Session()

def main():
    st.set_page_config(
        page_title="Fake news detector",
        page_icon=":star:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title(":newspaper: English fake news detector")
    sess = load_session()

#     model_names = ["RandomForest","MLP", "Logistic Regression","SVM"]
    model_dict = {
            "RandomForest": "models/RandomForest.pkl",
            "Logistic Regression": "models/Logistic.pkl",
            "Multilayer perceptron":'models/MLP.pkl',
            'Support vector machine': 'models/SVM.pkl' 
            }
    # resources_dir = "resources"

    for model_name, model_dir in model_dict.items():
        with open(model_dir, "rb") as f:
            model = pickle.load(f)
            model_dict[model_name] = model
        
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        tfidf_vectorizer = pickle.load(f)

    col1, col2 = st.columns([6, 4])
    with col2:
        st.image(f"https://previews.123rf.com/images/nlshop1/nlshop11703/nlshop1170300140/73523476-cartoon-characters-man-under-newspapers.jpg", width=700)
#     col1 = st.columns([6])
    with col1:
        model_name = st.selectbox("Choose your model", index=0, options=list(model_dict.keys()))

        news = st.text_area("Insert a piece of news here")
        entered_items = st.empty()

    button = st.button("Predict if this is real or fake!")

    st.markdown(
        "<hr />",
        unsafe_allow_html=True
    )

    if button:
        with st.spinner("Predicting..."):
            if not len(news):
                entered_items.markdown("In put at least a piece of news")
            else:
                model = model_dict[model_name]

                cleaned_news = text_preprocessing(news)
                
                cleaned_news_list = cleaned_news.split(" ")
                cleaned_news_dict = dict()
                for item in cleaned_news_list:
                    cleaned_news_dict[item] = cleaned_news_dict.get(item,0)
                    
                tfDict = {}
                wordsCount = sum(cleaned_news_dict.values())
                for word, count in cleaned_news_dict.items():
                    tfDict[word] = count/float(wordsCount + 0.0001)
                    
                idfDict = {}
                N = len(cleaned_news_list)
                
                text_vectorized = pd.DataFrame(tfDict)
                pred = model.predict(texxt_vectorized)[0]

#     for word, val in idfDict.items():
#         idfDict[word] = math.log10(N / (0.0000001+float(val)))
#     return idfDict
#                 text_vectorized = tfidf_vectorizer.transform([cleaned_news])
               
#                 pred = model.predict(text_vectorized[0,:3829])[0]

                if pred == 0:
                    st.markdown("Real news")
                else:
                    st.markdown("Fake news!!!")


if __name__ == "__main__":
    main()
