import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob
import joblib
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

# Load data
try:
    df = pd.read_csv('3-6-25_ai_dev_assignment_tickets_complex_1000.xls - OpTransactionHistoryUX3.csv')
except FileNotFoundError:
    print("File not found. Ensure it's in the same directory.")
    exit()

# Cleaning
# Fill missing values
df['ticket_text'] = df['ticket_text'].fillna('')
df['issue_type'] = df['issue_type'].fillna('Unknown')
df['urgency_level'] = df['urgency_level'].fillna('Unknown')
df['product'] = df['product'].fillna('Unknown')

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
    tokens = [w for w in tokens if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)

df['cleaned_text'] = df['ticket_text'].apply(preprocess_text)

# Filter training data
df_train = df[(df['issue_type'] != 'Unknown') & (df['urgency_level'] != 'Unknown')].copy()

# TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf_vectorizer.fit_transform(df_train['cleaned_text'])

# Numeric features
df_train['ticket_length'] = df_train['ticket_text'].apply(len)
df_train['sentiment_score'] = df_train['ticket_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
additional_features = df_train[['ticket_length', 'sentiment_score']].values
X_combined = hstack([X_tfidf, additional_features])

# Encode targets
le_issue_type = LabelEncoder()
df_train['issue_type_encoded'] = le_issue_type.fit_transform(df_train['issue_type'])
le_urgency_level = LabelEncoder()
df_train['urgency_level_encoded'] = le_urgency_level.fit_transform(df_train['urgency_level'])

# Issue type model
X_train_issue, X_test_issue, y_train_issue, y_test_issue = train_test_split(
    X_combined, df_train['issue_type_encoded'], test_size=0.2, random_state=42)
issue_type_classifier = LogisticRegression(max_iter=1000)
issue_type_classifier.fit(X_train_issue, y_train_issue)
y_pred_issue = issue_type_classifier.predict(X_test_issue)
print(f"Issue Type Classifier Accuracy: {accuracy_score(y_test_issue, y_pred_issue):.2f}")

# Urgency level model
X_train_urgency, X_test_urgency, y_train_urgency, y_test_urgency = train_test_split(
    X_combined, df_train['urgency_level_encoded'], test_size=0.2, random_state=42)
urgency_level_classifier = LogisticRegression(max_iter=1000)
urgency_level_classifier.fit(X_train_urgency, y_train_urgency)
y_pred_urgency = urgency_level_classifier.predict(X_test_urgency)
print(f"Urgency Level Classifier Accuracy: {accuracy_score(y_test_urgency, y_pred_urgency):.2f}")

# Save models
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(issue_type_classifier, 'issue_type_classifier.pkl')
joblib.dump(urgency_level_classifier, 'urgency_level_classifier.pkl')
joblib.dump(le_issue_type, 'le_issue_type.pkl')
joblib.dump(le_urgency_level, 'le_urgency_level.pkl')

# Entity extraction
product_list = df['product'].unique().tolist()
complaint_keywords = [
    'broken', 'late', 'error', 'faulty', 'damaged', 'not working', 'malfunction',
    'glitchy', 'missing', 'no response', 'stuck', 'failed', 'issue', 'problem',
    'charged', 'overbilled', 'underbilled', 'debited incorrectly', 'wrong item',
    'cannot log in', 'login not working', 'account problem', 'installation issue',
    'setup fails'
]

def extract_entities(text):
    entities = {"product_names": [], "dates": [], "complaint_keywords": []}
    for product in product_list:
        if product != 'Unknown' and re.search(r'\b' + re.escape(product.lower()) + r'\b', text.lower()):
            entities["product_names"].append(product)
    date_patterns = [
        r'\b\d{1,2} (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?: \d{4})?\b',
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
        r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',
        r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b'
    ]
    for pattern in date_patterns:
        entities["dates"].extend(re.findall(pattern, text, re.IGNORECASE))
    for keyword in complaint_keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', text.lower()):
            entities["complaint_keywords"].append(keyword)
    entities["product_names"] = list(set(entities["product_names"]))
    entities["dates"] = list(set(entities["dates"]))
    entities["complaint_keywords"] = list(set(entities["complaint_keywords"]))
    return entities

# Load models
loaded_tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
loaded_issue_type_classifier = joblib.load('issue_type_classifier.pkl')
loaded_urgency_level_classifier = joblib.load('urgency_level_classifier.pkl')
loaded_le_issue_type = joblib.load('le_issue_type.pkl')
loaded_le_urgency_level = joblib.load('le_urgency_level.pkl')

# Prediction + extraction
def classify_and_extract(raw_ticket_text):
    cleaned_text = preprocess_text(raw_ticket_text)
    text_tfidf = loaded_tfidf_vectorizer.transform([cleaned_text])
    ticket_length = len(raw_ticket_text)
    sentiment_score = TextBlob(raw_ticket_text).sentiment.polarity
    input_features = hstack([text_tfidf, [[ticket_length, sentiment_score]]])
    issue_pred = loaded_issue_type_classifier.predict(input_features)
    urgency_pred = loaded_urgency_level_classifier.predict(input_features)
    return {
        "predicted_issue_type": loaded_le_issue_type.inverse_transform(issue_pred)[0],
        "predicted_urgency_level": loaded_le_urgency_level.inverse_transform(urgency_pred)[0],
        "extracted_entities": extract_entities(raw_ticket_text)
    }

# Example predictions
print("--- Running Example Predictions ---")
ticket_example_1 = "My SmartWatch V2 broke after 2 days. The screen is completely black. Order #12345 placed on 15 May 2023. I need urgent help."
print("Example 1 Results:", classify_and_extract(ticket_example_1))

ticket_example_2 = "I cannot log in to my account since yesterday, 01/06/2024. My password is not working. The product is Vision LED TV."
print("Example 2 Results:", classify_and_extract(ticket_example_2))

ticket_example_3 = "The delivery for my EcoBreeze AC, order #98765, is very late. It was supposed to arrive on 25-04-2024."
print("Example 3 Results:", classify_and_extract(ticket_example_3))

ticket_example_4 = "Can you tell me more about the FitRun Treadmill warranty? Is it available in black?"
print("Example 4 Results:", classify_and_extract(ticket_example_4))

ticket_example_5 = "I was incorrectly charged twice for order #12345 for my ProTab X1 on 03 March."
print("Example 5 Results:", classify_and_extract(ticket_example_5))
print("---------------------------------")
