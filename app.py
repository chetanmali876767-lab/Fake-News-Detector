from pathlib import Path

import joblib
import streamlit as st


MODEL_PATH = Path(__file__).with_name("fake_news_model.pkl")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("Fake News Detector")
st.write("Enter a headline and/or article text for a prediction.")

model = load_model()

title = st.text_input("Headline")
article_text = st.text_area("Article text", height=250)

if st.button("Check article"):
    content = f"{title.strip()} {article_text.strip()}".strip()

    if not content:
        st.warning("Enter a headline or article text first.")
    else:
        prediction = int(model.predict([content])[0])
        probabilities = model.predict_proba([content])[0]
        confidence = float(max(probabilities)) * 100

        if prediction == 1:
            st.error("Prediction: Likely fake news")
        else:
            st.success("Prediction: Likely real news")

        st.metric("Model confidence", f"{confidence:.1f}%")
        st.caption("This is a model prediction, not a guarantee that the article is true.")