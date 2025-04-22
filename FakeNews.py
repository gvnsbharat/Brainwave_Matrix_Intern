
# === IMPORT LIBRARIES ===
import pandas as pd
import numpy as np
import re
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# === LOAD DATA ===
fake_df = pd.read_csv("/Fake.csv", usecols=['text'])
true_df = pd.read_csv("/True.csv", usecols=['text'])
fake_df['label'] = 1
true_df['label'] = 0
df = pd.concat([fake_df, true_df], axis=0).sample(frac=1).reset_index(drop=True)

# === TEXT PREPROCESSING ===
stop_words = stopwords.words('english')
ps = PorterStemmer()

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)

df['text'] = df['text'].apply(preprocess_text)

# === TRAIN TEST SPLIT ===
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === TF-IDF & MODEL TRAINING ===
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# === METRICS ===
y_pred = model.predict(X_test_tfidf)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))

# === PREDICT FUNCTION ===
def predict_news(news_text):
    cleaned = preprocess_text(news_text)
    tfidf_input = vectorizer.transform([cleaned])
    prediction = model.predict(tfidf_input)[0]
    return "🟢 Real News" if prediction == 0 else "🔴 Fake News"

# === GRADIO BLOCKS INTERFACE ===
with gr.Blocks(title="Fake News Detector") as demo:
    gr.Markdown(
        """
        # 📰 Fake News Detector
        Paste a news article below and click **Detect** to check if it's Fake or Real.
        """
    )
    
    with gr.Row():
        with gr.Column():
            news_input = gr.Textbox(label="News Article", lines=10, placeholder="Type or paste your news text here...")
            detect_button = gr.Button("🔍 Detect")
        with gr.Column():
            result_output = gr.Textbox(label="Prediction Result", interactive=False)

    detect_button.click(fn=predict_news, inputs=news_input, outputs=result_output)

# === LAUNCH APP ===
demo.launch(share=True)
