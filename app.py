import gradio as gr
import pandas as pd
import re, string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob
import joblib
from scipy.sparse import hstack
import warnings
from datetime import datetime
from dateutil.parser import parse

warnings.filterwarnings('ignore')

# Load product list
try:
    df_products = pd.read_csv('3-6-25_ai_dev_assignment_tickets_complex_1000.xls - OpTransactionHistoryUX3.csv')
    df_products['product'] = df_products['product'].fillna('Unknown')
    product_list = df_products['product'].unique().tolist()
except FileNotFoundError:
    print("Product list CSV not found.")
    product_list = []

# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Complaint keywords
complaint_keywords = ['broken', 'late', 'error', 'faulty', 'damaged', 'not working', 'malfunction',
                      'glitchy', 'missing', 'no response', 'stuck', 'failed', 'issue', 'problem',
                      'charged', 'overbilled', 'underbilled', 'debited incorrectly', 'wrong item',
                      'cannot log in', 'login not working', 'account problem', 'installation issue',
                      'setup fails']

# Entity extraction
def extract_entities(text):
    extracted_entities = {"product_names": [], "dates": [], "complaint_keywords": []}
    for product in product_list:
        if product != 'Unknown' and re.search(r'\b' + re.escape(product.lower()) + r'\b', text.lower()):
            extracted_entities["product_names"].append(product)

    date_patterns = [
        r'\b\d{1,2} (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?: \d{4})?\b',
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
        r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',
        r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b'
    ]

    found_raw_dates = []
    for pattern in date_patterns:
        found_raw_dates.extend(re.findall(pattern, text, re.IGNORECASE))

    normalized_dates = []
    current_year = datetime.now().year
    for date_str in set(found_raw_dates):
        try:
            parsed_date = parse(date_str, fuzzy=True)
            year_present = bool(re.search(r'\d{4}|\d{2}(?=\b)', date_str))
            normalized_dates.append(parsed_date.strftime('%d/%m/%Y') if year_present else parsed_date.strftime('%d/%m/--'))
        except (ValueError, OverflowError):
            normalized_dates.append(date_str)

    extracted_entities["dates"] = list(set(normalized_dates))

    for keyword in complaint_keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', text.lower()):
            extracted_entities["complaint_keywords"].append(keyword)

    extracted_entities["product_names"] = list(set(extracted_entities["product_names"]))
    extracted_entities["complaint_keywords"] = list(set(extracted_entities["complaint_keywords"]))

    return extracted_entities

# Load trained models
# joblib loading
try:
    loaded_tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    loaded_issue_type_classifier = joblib.load('issue_type_classifier.pkl')
    loaded_urgency_level_classifier = joblib.load('urgency_level_classifier.pkl')
    loaded_le_issue_type = joblib.load('le_issue_type.pkl')
    loaded_le_urgency_level = joblib.load('le_urgency_level.pkl')
except FileNotFoundError:
    print("Model file(s) missing.")
    exit()

# Main function for prediction and extraction
def classify_and_extract_for_gradio(raw_ticket_text):
    if not raw_ticket_text:
        return "Please enter some ticket text.", "N/A", "N/A", "N/A", "N/A"

    cleaned_text = preprocess_text(raw_ticket_text)
    text_tfidf = loaded_tfidf_vectorizer.transform([cleaned_text])
    ticket_length = len(raw_ticket_text)
    sentiment_score = TextBlob(raw_ticket_text).sentiment.polarity
    input_features_combined = hstack([text_tfidf, [[ticket_length, sentiment_score]]])

    issue_type_pred = loaded_issue_type_classifier.predict(input_features_combined)
    urgency_level_pred = loaded_urgency_level_classifier.predict(input_features_combined)

    predicted_issue_type = loaded_le_issue_type.inverse_transform(issue_type_pred)[0]
    predicted_urgency_level = loaded_le_urgency_level.inverse_transform(urgency_level_pred)[0]

    entities = extract_entities(raw_ticket_text)

    product_names_str = ", ".join(entities["product_names"]) if entities["product_names"] else "None"
    dates_str = ", ".join(entities["dates"]) if entities["dates"] else "None"
    complaint_keywords_str = ", ".join(entities["complaint_keywords"]) if entities["complaint_keywords"] else "None"

    return (f"**Issue Type:** {predicted_issue_type}",
            f"**Urgency Level:** {predicted_urgency_level}",
            f"**Products:** {product_names_str}",
            f"**Dates:** {dates_str}",
            f"**Complaint Keywords:** {complaint_keywords_str}")

# Gradio UI
iface = gr.Interface(
    fn=classify_and_extract_for_gradio,
    inputs=gr.Textbox(lines=5, label="Enter Customer Support Ticket Text Here"),
    outputs=[
        gr.Markdown(label="Predicted Issue Type"),
        gr.Markdown(label="Predicted Urgency Level"),
        gr.Markdown(label="Extracted Product Names"),
        gr.Markdown(label="Extracted Dates"),
        gr.Markdown(label="Extracted Complaint Keywords")
    ],
    title="Customer Support Ticket Analysis",
    description="Enter a customer support ticket to get predictions and entity extraction.",
    examples=[]
)

# Launch
if __name__ == "__main__":
    iface.launch()
