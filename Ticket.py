import pandas as pd # Import pandas for data manipulation (e.g., reading CSV, handling DataFrames).
import re # Import re for regular expressions, used in text cleaning and entity extraction.
import string # Import string to access punctuation characters.
from nltk.corpus import stopwords # Import stopwords from NLTK for removing common words.
from nltk.stem import WordNetLemmatizer # Import WordNetLemmatizer from NLTK for lemmatization.
from nltk.tokenize import word_tokenize # Import word_tokenize from NLTK for breaking text into words.
from sklearn.feature_extraction.text import TfidfVectorizer # Import TfidfVectorizer for creating text features.
from sklearn.model_selection import train_test_split # Import train_test_split for dividing data into training and testing sets.
from sklearn.linear_model import LogisticRegression # Import LogisticRegression, our chosen classification model.
from sklearn.preprocessing import LabelEncoder # Import LabelEncoder for converting text labels to numbers.
from textblob import TextBlob # Import TextBlob for sentiment analysis.
import joblib # Import joblib for saving and loading Python objects (our models).
from scipy.sparse import hstack # Import hstack for combining sparse matrices (TF-IDF) with other features.
from sklearn.metrics import accuracy_score # Import accuracy_score to evaluate model performance.
import warnings # Import warnings to manage warning messages.

# Suppress all warnings for cleaner output in the console.
warnings.filterwarnings('ignore')

# --- 1. Data Preparation ---

# Load the dataset from the CSV file.
try:
    # Attempt to read the CSV file into a pandas DataFrame.
    df = pd.read_csv('3-6-25_ai_dev_assignment_tickets_complex_1000.xls - OpTransactionHistoryUX3.csv')
except FileNotFoundError:
    # If the file is not found, print an error message and exit the program.
    print("Error: The file '3-6-25_ai_dev_assignment_tickets_complex_1000.xls - OpTransactionHistoryUX3.csv' was not found.")
    print("Please ensure it's in the same directory as the Python script.")
    exit()

# Handle missing data (NaN values) in specific columns.
# Fill empty cells in 'ticket_text' with an empty string.
df['ticket_text'] = df['ticket_text'].fillna('')
# Fill empty cells in 'issue_type', 'urgency_level', and 'product' with 'Unknown'.
df['issue_type'] = df['issue_type'].fillna('Unknown')
df['urgency_level'] = df['urgency_level'].fillna('Unknown')
df['product'] = df['product'].fillna('Unknown')

# Define a function to preprocess text for machine learning.
def preprocess_text(text):
    text = text.lower() # Convert all text to lowercase to ensure consistency.
    text = re.sub(r'\[.*?\]', '', text) # Remove any text enclosed in square brackets (e.g., "[contact info]").
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs (web addresses).
    text = re.sub(r'<.*?>+', '', text) # Remove HTML tags (e.g., <br>, <div>).
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text) # Remove all punctuation marks (e.g., periods, commas).
    text = re.sub(r'\n', '', text) # Remove newline characters.
    text = re.sub(r'\w*\d\w*', '', text) # Remove words that contain numbers (e.g., "order123", "v2").
    tokens = word_tokenize(text) # Break the text into individual words (tokens).
    stop_words = set(stopwords.words('english')) # Get a list of common English stopwords (e.g., "the", "is", "a").
    tokens = [word for word in tokens if word not in stop_words] # Remove stopwords from the list of tokens.
    lemmatizer = WordNetLemmatizer() # Initialize the WordNet Lemmatizer to reduce words to their base form.
    tokens = [lemmatizer.lemmatize(word) for word in tokens] # Apply lemmatization to each token.
    return ' '.join(tokens) # Join the processed tokens back into a single string, separated by spaces.

# Apply the `preprocess_text` function to the 'ticket_text' column.
# This creates a new column 'cleaned_text' with the processed content.
df['cleaned_text'] = df['ticket_text'].apply(preprocess_text)

# Create a new DataFrame `df_train` that will be used for training our models.
# We filter out rows where 'issue_type' or 'urgency_level' are 'Unknown' because these are missing labels.
df_train = df[df['issue_type'] != 'Unknown'].copy() # Keep only rows with a known issue type.
df_train = df_train[df_train['urgency_level'] != 'Unknown'].copy() # Keep only rows with a known urgency level.

# --- 2. Feature Engineering ---

# Initialize a TF-IDF Vectorizer.
# TF-IDF converts text into numerical features, representing the importance of words.
tfidf_vectorizer = TfidfVectorizer(max_features=1000) # `max_features=1000` means we'll only use the 1000 most important words as features.
# Fit the vectorizer to our cleaned training text and transform the text into TF-IDF features.
X_tfidf = tfidf_vectorizer.fit_transform(df_train['cleaned_text'])

# Extract 'ticket_length' as a numerical feature.
# This measures how many characters are in the original ticket text.
df_train['ticket_length'] = df_train['ticket_text'].apply(len)

# Extract 'sentiment_score' as a numerical feature using TextBlob.
# Sentiment polarity ranges from -1 (very negative) to 1 (very positive).
df_train['sentiment_score'] = df_train['ticket_text'].apply(lambda text: TextBlob(text).sentiment.polarity)

# Combine the TF-IDF features with the additional numerical features.
# `hstack` is used to horizontally stack (combine side-by-side) the sparse TF-IDF matrix with our dense numerical features.
additional_features = df_train[['ticket_length', 'sentiment_score']].values # Get the values of ticket_length and sentiment_score.
X_combined = hstack([X_tfidf, additional_features]) # X_combined is now our complete feature set for training.

# --- 3. Multi-Task Learning (Model Training) ---

# Initialize LabelEncoders for our categorical target variables.
# LabelEncoder converts text labels (e.g., 'Billing Problem', 'Low') into numerical labels (e.g., 0, 1, 2).
le_issue_type = LabelEncoder() # Create an encoder for issue types.
df_train['issue_type_encoded'] = le_issue_type.fit_transform(df_train['issue_type']) # Apply encoding to issue types.
le_urgency_level = LabelEncoder() # Create an encoder for urgency levels.
df_train['urgency_level_encoded'] = le_urgency_level.fit_transform(df_train['urgency_level']) # Apply encoding to urgency levels.

# --- Issue Type Classifier Training ---
# Prepare data for the Issue Type Classifier.
X_issue = X_combined # Our combined features are the input (X) for the model.
y_issue = df_train['issue_type_encoded'] # The encoded issue types are the target (y) for the model.
# Split the data into training and testing sets. 80% for training, 20% for testing.
# `random_state=42` ensures that the split is always the same for reproducibility.
X_train_issue, X_test_issue, y_train_issue, y_test_issue = train_test_split(X_issue, y_issue, test_size=0.2, random_state=42)

# Train the Issue Type Classifier.
issue_type_classifier = LogisticRegression(max_iter=1000) # Initialize Logistic Regression model. `max_iter` increased for convergence.
issue_type_classifier.fit(X_train_issue, y_train_issue) # Train the model using the training data.

# Evaluate the Issue Type Classifier's performance.
y_pred_issue = issue_type_classifier.predict(X_test_issue) # Make predictions on the unseen test data.
accuracy_issue = accuracy_score(y_test_issue, y_pred_issue) # Calculate the accuracy of the predictions.
print(f"Issue Type Classifier Accuracy: {accuracy_issue:.2f}") # Print the accuracy, formatted to two decimal places.

# --- Urgency Level Classifier Training ---
# Prepare data for the Urgency Level Classifier.
X_urgency = X_combined # Our combined features are the input (X) for this model too.
y_urgency = df_train['urgency_level_encoded'] # The encoded urgency levels are the target (y).
# Split the data into training and testing sets, similar to the issue type classifier.
X_train_urgency, X_test_urgency, y_train_urgency, y_test_urgency = train_test_split(X_urgency, y_urgency, test_size=0.2, random_state=42)

# Train the Urgency Level Classifier.
urgency_level_classifier = LogisticRegression(max_iter=1000) # Initialize another Logistic Regression model.
urgency_level_classifier.fit(X_train_urgency, y_train_urgency) # Train this model using its training data.

# Evaluate the Urgency Level Classifier's performance.
y_pred_urgency = urgency_level_classifier.predict(X_test_urgency) # Make predictions on the unseen test data.
accuracy_urgency = accuracy_score(y_test_urgency, y_pred_urgency) # Calculate the accuracy.
print(f"Urgency Level Classifier Accuracy: {accuracy_urgency:.2f}") # Print the accuracy.

# Save the trained models and encoders to disk.
# This is crucial so we don't have to retrain them every time we want to make a prediction.
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl') # Save the TF-IDF vectorizer.
joblib.dump(issue_type_classifier, 'issue_type_classifier.pkl') # Save the trained issue type model.
joblib.dump(urgency_level_classifier, 'urgency_level_classifier.pkl') # Save the trained urgency level model.
joblib.dump(le_issue_type, 'le_issue_type.pkl') # Save the issue type label encoder.
joblib.dump(le_urgency_level, 'le_urgency_level.pkl') # Save the urgency level label encoder.

# --- 4. Entity Extraction ---

# Define lists for entity extraction.
# Get unique product names directly from the loaded dataset.
product_list = df['product'].unique().tolist()
# Define a list of common complaint keywords.
complaint_keywords = ['broken', 'late', 'error', 'faulty', 'damaged', 'not working', 'malfunction',
                      'glitchy', 'missing', 'no response', 'stuck', 'failed', 'issue', 'problem',
                      'charged', 'overbilled', 'underbilled', 'debited incorrectly', 'wrong item',
                      'cannot log in', 'login not working', 'account problem', 'installation issue',
                      'setup fails']

# Define a function to extract key entities (product names, dates, complaint keywords) from raw text.
def extract_entities(text):
    extracted_entities = {
        "product_names": [], # Initialize an empty list for product names.
        "dates": [], # Initialize an empty list for dates.
        "complaint_keywords": [] # Initialize an empty list for complaint keywords.
    }

    # Extract product names: Iterate through our known product list.
    for product in product_list:
        # If the product is not 'Unknown' and its lowercase form is found in the lowercase text (as a whole word).
        if product != 'Unknown' and re.search(r'\b' + re.escape(product.lower()) + r'\b', text.lower()):
            extracted_entities["product_names"].append(product) # Add the found product to the list.

    # Define various regular expression patterns to find dates.
    date_patterns = [
        r'\b\d{1,2} (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?: \d{4})?\b', # Matches "15 May 2023" or "03 March".
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', # Matches "01/06/2024" or "1/6/24".
        r'\b\d{1,2}-\d{1,2}-\d{2,4}\b', # Matches "25-04-2024" or "5-4-24".
        r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b' # Matches "March 3rd, 2023".
    ]
    # Iterate through each date pattern.
    for pattern in date_patterns:
        found_dates = re.findall(pattern, text, re.IGNORECASE) # Find all dates matching the current pattern (case-insensitive).
        extracted_entities["dates"].extend(found_dates) # Add all found dates to the list.

    # Extract complaint keywords: Iterate through our predefined list of keywords.
    for keyword in complaint_keywords:
        # If the lowercase keyword is found in the lowercase text (as a whole word).
        if re.search(r'\b' + re.escape(keyword) + r'\b', text.lower()):
            extracted_entities["complaint_keywords"].append(keyword) # Add the found keyword to the list.

    # Remove duplicate entries from each extracted entity list.
    extracted_entities["product_names"] = list(set(extracted_entities["product_names"]))
    extracted_entities["dates"] = list(set(extracted_entities["dates"]))
    extracted_entities["complaint_keywords"] = list(set(extracted_entities["complaint_keywords"]))

    return extracted_entities # Return the dictionary containing all extracted entities.

# --- 5. Integration ---

# Load the pre-trained models and encoders from the .pkl files.
# These files were saved in the training phase and are now loaded for making predictions.
loaded_tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl') # Load the TF-IDF vectorizer.
loaded_issue_type_classifier = joblib.load('issue_type_classifier.pkl') # Load the issue type classifier.
loaded_urgency_level_classifier = joblib.load('urgency_level_classifier.pkl') # Load the urgency level classifier.
loaded_le_issue_type = joblib.load('le_issue_type.pkl') # Load the issue type label encoder.
loaded_le_urgency_level = joblib.load('le_urgency_level.pkl') # Load the urgency level label encoder.

# Define the main function that takes raw ticket text and returns predictions and extracted entities.
def classify_and_extract(raw_ticket_text):
    # Preprocess the raw input text.
    cleaned_text = preprocess_text(raw_ticket_text)

    # Perform feature engineering on the preprocessed text for prediction.
    text_tfidf = loaded_tfidf_vectorizer.transform([cleaned_text]) # Transform the cleaned text using the loaded TF-IDF vectorizer.
    ticket_length = len(raw_ticket_text) # Calculate the length of the raw ticket text.
    sentiment_score = TextBlob(raw_ticket_text).sentiment.polarity # Calculate the sentiment polarity.

    # Combine TF-IDF features with the additional numerical features.
    input_features_combined = hstack([text_tfidf, [[ticket_length, sentiment_score]]])

    # Predict the issue type.
    issue_type_prediction_encoded = loaded_issue_type_classifier.predict(input_features_combined) # Get the numerical prediction.
    predicted_issue_type = loaded_le_issue_type.inverse_transform(issue_type_prediction_encoded)[0] # Convert numerical prediction back to text label.

    # Predict the urgency level.
    urgency_level_prediction_encoded = loaded_urgency_level_classifier.predict(input_features_combined) # Get the numerical prediction.
    predicted_urgency_level = loaded_le_urgency_level.inverse_transform(urgency_level_prediction_encoded)[0] # Convert numerical prediction back to text label.

    # Extract entities from the raw ticket text.
    extracted_entities = extract_entities(raw_ticket_text)

    # Return all results in a dictionary.
    return {
        "predicted_issue_type": predicted_issue_type,
        "predicted_urgency_level": predicted_urgency_level,
        "extracted_entities": extracted_entities
    }

# --- Example Usage (These lines will run when you execute Ticket.py) ---

print("--- Running Example Predictions ---")
# Example 1: A product defect and late delivery scenario.
ticket_example_1 = "My SmartWatch V2 broke after 2 days. The screen is completely black. Order #12345 placed on 15 May 2023. I need urgent help."
results_1 = classify_and_extract(ticket_example_1)
print("Example 1 Results:", results_1)

# Example 2: An account access issue with a date.
ticket_example_2 = "I cannot log in to my account since yesterday, 01/06/2024. My password is not working. The product is Vision LED TV."
results_2 = classify_and_extract(ticket_example_2)
print("Example 2 Results:", results_2)

# Example 3: A late delivery complaint with a specific date format.
ticket_example_3 = "The delivery for my EcoBreeze AC, order #98765, is very late. It was supposed to arrive on 25-04-2024."
results_3 = classify_and_extract(ticket_example_3)
print("Example 3 Results:", results_3)

# Example 4: A general inquiry about a product.
ticket_example_4 = "Can you tell me more about the FitRun Treadmill warranty? Is it available in black?"
results_4 = classify_and_extract(ticket_example_4)
print("Example 4 Results:", results_4)

# Example 5: A billing problem.
ticket_example_5 = "I was incorrectly charged twice for order #12345 for my ProTab X1 on 03 March."
results_5 = classify_and_extract(ticket_example_5)
print("Example 5 Results:", results_5)
print("---------------------------------")
