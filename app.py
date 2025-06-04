import gradio as gr # Import the Gradio library for building the web interface.
import pandas as pd # Import pandas for data manipulation (specifically for loading product list).
import re # Import re for regular expressions, used in text cleaning and entity extraction.
import string # Import string to access punctuation characters.
from nltk.corpus import stopwords # Import stopwords from NLTK.
from nltk.stem import WordNetLemmatizer # Import WordNetLemmatizer from NLTK.
from nltk.tokenize import word_tokenize # Import word_tokenize from NLTK.
from sklearn.feature_extraction.text import TfidfVectorizer # Import TfidfVectorizer.
from sklearn.linear_model import LogisticRegression # Import LogisticRegression.
from sklearn.preprocessing import LabelEncoder # Import LabelEncoder.
from textblob import TextBlob # Import TextBlob for sentiment analysis.
import joblib # Import joblib for loading saved models.
from scipy.sparse import hstack # Import hstack for combining features.
import warnings # Import warnings to manage warning messages.
from datetime import datetime # Import datetime to work with dates.
from dateutil.parser import parse # Import parse from dateutil for flexible date parsing.

# Suppress all warnings for cleaner output in the Gradio app's console.
warnings.filterwarnings('ignore')

# --- Re-define necessary functions and load models for the Gradio app ---
# These functions and model loading steps are repeated here to make the Gradio app self-contained.
# This means 'app.py' can run independently after 'Ticket.py' has created the .pkl files.

# Load the dataset (or at least the product column) to get the product list for entity extraction.
try:
    # Attempt to read the CSV file to get product names.
    df_products = pd.read_csv('3-6-25_ai_dev_assignment_tickets_complex_1000.xls - OpTransactionHistoryUX3.csv')
    df_products['product'] = df_products['product'].fillna('Unknown')
    product_list = df_products['product'].unique().tolist() # Get a list of all unique product names.
except FileNotFoundError:
    # If the product list CSV is not found, print an error and set an empty product list.
    # Entity extraction for products will not work correctly without this file.
    print("Error: Product list CSV not found for Gradio app. Entity extraction for products may be limited.")
    product_list = [] # Fallback to an empty list.

# Define the preprocessing function (copied from the main script).
def preprocess_text(text):
    text = text.lower() # Convert text to lowercase.
    text = re.sub(r'\[.*?\]', '', text) # Remove text in square brackets.
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs.
    text = re.sub(r'<.*?>+', '', text) # Remove HTML tags.
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text) # Remove punctuation.
    text = re.sub(r'\n', '', text) # Remove newline characters.
    text = re.sub(r'\w*\d\w*', '', text) # Remove words containing numbers.
    tokens = word_tokenize(text) # Tokenize the text.
    stop_words = set(stopwords.words('english')) # Define English stopwords.
    tokens = [word for word in tokens if word not in stop_words] # Remove stopwords.
    lemmatizer = WordNetLemmatizer() # Initialize lemmatizer.
    tokens = [lemmatizer.lemmatize(word) for word in tokens] # Lemmatize tokens.
    return ' '.join(tokens) # Join tokens back into a string.

# Define complaint keywords for entity extraction (copied from the main script).
complaint_keywords = ['broken', 'late', 'error', 'faulty', 'damaged', 'not working', 'malfunction',
                      'glitchy', 'missing', 'no response', 'stuck', 'failed', 'issue', 'problem',
                      'charged', 'overbilled', 'underbilled', 'debited incorrectly', 'wrong item',
                      'cannot log in', 'login not working', 'account problem', 'installation issue',
                      'setup fails']

# Define the entity extraction function (copied from the main script).
def extract_entities(text):
    extracted_entities = {
        "product_names": [],
        "dates": [],
        "complaint_keywords": []
    }

    # Extract product names by searching for known products in the text.
    for product in product_list:
        if product != 'Unknown' and re.search(r'\b' + re.escape(product.lower()) + r'\b', text.lower()):
            extracted_entities["product_names"].append(product)

    # Define various regex patterns to extract dates.
    # These patterns are designed to capture common date formats.
    date_patterns = [
        r'\b\d{1,2} (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?: \d{4})?\b',
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
        r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',
        r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b'
    ]

    found_raw_dates = [] # Temporarily store raw date strings.
    for pattern in date_patterns:
        found_raw_dates.extend(re.findall(pattern, text, re.IGNORECASE))

    normalized_dates = []
    current_year = datetime.now().year # Get the current year to use as a default if year is missing.

    for date_str in set(found_raw_dates): # Process unique raw dates.
        try:
            # Try to parse the date string.
            # If a year is not explicitly mentioned, parse will default to the current year.
            parsed_date = parse(date_str, fuzzy=True)

            # Check if the year was explicitly mentioned in the original string.
            # This is a heuristic: if the original string contains a 4-digit number that looks like a year,
            # or if it contains a 2-digit number that could be a year in a DD/MM/YY format, assume year is present.
            # Otherwise, we might consider it missing for the purpose of formatting.
            year_present_in_string = bool(re.search(r'\d{4}|\d{2}(?=\b)', date_str))

            if year_present_in_string:
                normalized_dates.append(parsed_date.strftime('%d/%m/%Y')) # Format as DD/MM/YYYY.
            else:
                # If year is not clearly present, format as DD/MM/--.
                normalized_dates.append(parsed_date.strftime('%d/%m/--'))

        except ValueError:
            # If parsing fails, keep the original string or skip it.
            # For this case, we'll keep the original string to avoid losing information.
            normalized_dates.append(date_str)
        except OverflowError:
            # Handle cases where parsing might lead to an overflow (e.g., extremely large year).
            normalized_dates.append(date_str)


    extracted_entities["dates"] = list(set(normalized_dates)) # Store unique normalized dates.

    # Extract complaint keywords by searching for defined keywords in the text.
    for keyword in complaint_keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', text.lower()):
            extracted_entities["complaint_keywords"].append(keyword)

    # Remove duplicate entries from each extracted entity list.
    extracted_entities["product_names"] = list(set(extracted_entities["product_names"]))
    extracted_entities["complaint_keywords"] = list(set(extracted_entities["complaint_keywords"]))

    return extracted_entities

# Load the pre-trained models and encoders that were saved by Ticket.py.
# This is a critical step for the Gradio app to function, as it relies on these trained components.
try:
    loaded_tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl') # Load the saved TF-IDF vectorizer.
    loaded_issue_type_classifier = joblib.load('issue_type_classifier.pkl') # Load the saved issue type classifier model.
    loaded_urgency_level_classifier = joblib.load('urgency_level_classifier.pkl') # Load the saved urgency level classifier model.
    loaded_le_issue_type = joblib.load('le_issue_type.pkl') # Load the saved issue type label encoder.
    loaded_le_urgency_level = joblib.load('le_urgency_level.pkl') # Load the saved urgency level label encoder.
except FileNotFoundError:
    # If any model file is not found, print an error and exit. The app cannot run without these.
    print("Error: One or more .pkl model files not found.")
    print("Please ensure 'tfidf_vectorizer.pkl', 'issue_type_classifier.pkl', 'urgency_level_classifier.pkl',")
    print("'le_issue_type.pkl', and 'le_urgency_level.pkl' are in the same directory.")
    exit() # Exit the program if models cannot be loaded.

# Define the main function that Gradio will call to process user input.
def classify_and_extract_for_gradio(raw_ticket_text):
    # Check if the input text is empty.
    if not raw_ticket_text:
        # If empty, return placeholder messages.
        return "Please enter some ticket text.", "N/A", "N/A", "N/A", "N/A"

    # Preprocess the raw input text using the defined function.
    cleaned_text = preprocess_text(raw_ticket_text)

    # Transform the cleaned text into numerical features using the loaded TF-IDF vectorizer.
    text_tfidf = loaded_tfidf_vectorizer.transform([cleaned_text])
    # Calculate the length of the original raw text.
    ticket_length = len(raw_ticket_text)
    # Calculate the sentiment polarity of the original raw text.
    sentiment_score = TextBlob(raw_ticket_text).sentiment.polarity

    # Combine all features (TF-IDF and additional numerical features) for model prediction.
    input_features_combined = hstack([text_tfidf, [[ticket_length, sentiment_score]]])

    # Predict the issue type using the loaded classifier and decode the numerical prediction.
    issue_type_prediction_encoded = loaded_issue_type_classifier.predict(input_features_combined)
    predicted_issue_type = loaded_le_issue_type.inverse_transform(issue_type_prediction_encoded)[0]

    # Predict the urgency level using the loaded classifier and decode the numerical prediction.
    urgency_level_prediction_encoded = loaded_urgency_level_classifier.predict(input_features_combined)
    predicted_urgency_level = loaded_le_urgency_level.inverse_transform(urgency_level_prediction_encoded)[0]

    # Extract entities from the raw ticket text.
    extracted_entities = extract_entities(raw_ticket_text)

    # Format the extracted entities into readable strings for display in the Gradio interface.
    product_names_str = ", ".join(extracted_entities["product_names"]) if extracted_entities["product_names"] else "None"
    dates_str = ", ".join(extracted_entities["dates"]) if extracted_entities["dates"] else "None"
    complaint_keywords_str = ", ".join(extracted_entities["complaint_keywords"]) if extracted_entities["complaint_keywords"] else "None"

    # Return the formatted results. Gradio will display each item in a separate output box.
    return (f"**Issue Type:** {predicted_issue_type}", # Output 1: Predicted Issue Type (formatted with Markdown bold).
            f"**Urgency Level:** {predicted_urgency_level}", # Output 2: Predicted Urgency Level.
            f"**Products:** {product_names_str}", # Output 3: Extracted Product Names.
            f"**Dates:** {dates_str}", # Output 4: Extracted Dates.
            f"**Complaint Keywords:** {complaint_keywords_str}") # Output 5: Extracted Complaint Keywords.

# Create the Gradio interface.
iface = gr.Interface(
    fn=classify_and_extract_for_gradio, # The Python function to call when the user interacts.
    inputs=gr.Textbox(lines=5, label="Enter Customer Support Ticket Text Here"), # Input component: a multi-line text box.
    outputs=[ # List of output components to display the results.
        gr.Markdown(label="Predicted Issue Type"), # Markdown component for formatted text.
        gr.Markdown(label="Predicted Urgency Level"),
        gr.Markdown(label="Extracted Product Names"),
        gr.Markdown(label="Extracted Dates"),
        gr.Markdown(label="Extracted Complaint Keywords")
    ],
    title="Customer Support Ticket Analysis", # Title of the Gradio web app.
    description="Enter a customer support ticket to get its predicted issue type, urgency level, and extracted key entities.", # Description displayed below the title.
    examples=[] # Removed all examples as per your request.
)

# Launch the Gradio app.
if __name__ == "__main__":
    # This ensures the app only launches when the script is run directly (not when imported as a module).
    iface.launch() # Starts the Gradio web server and opens the app in your browser.
