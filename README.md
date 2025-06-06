# NLP-Ticket-Analysis
Customer Support Ticket Analysis and Classification

This project develops a machine learning pipeline to classify customer support tickets by their issue type and urgency level, and to extract key entities such as product names, dates, and complaint keywords. It also provides an interactive web interface using Gradio for easy interaction.
Project Structure

The project consists of the following key files:

    Ticket.py: The core Python script responsible for data preparation, feature engineering, training the machine learning models, and saving them to disk.

    app.py: The Python script that launches the interactive Gradio web application, loading the pre-trained models to make predictions on new ticket text.

    3-6-25_ai_dev_assignment_tickets_complex_1000.xls - OpTransactionHistoryUX3.csv: The anonymized customer support ticket dataset used for training the models.

    .pkl files (generated after running Ticket.py):

        tfidf_vectorizer.pkl: The saved TF-IDF vectorizer.

        issue_type_classifier.pkl: The trained model for classifying issue types.

        urgency_level_classifier.pkl: The trained model for classifying urgency levels.

        le_issue_type.pkl: The label encoder for issue types.

        le_urgency_level.pkl: The label encoder for urgency levels.

Setup:

Follow these steps to set up and run the project on your local machine.
1. Prerequisites

    Python 3.7/+

2. Clone/Download the Project

    If you have a Git repository, clone it. Otherwise, download all the project files (Ticket.py, app.py, and the CSV file) into a single folder on your computer.

3. Install Required Python Libraries

pip install pandas scikit-learn nltk textblob joblib gradio

4. Download NLTK Data

NLTK requires additional data for text processing (like stopwords and lemmatization data).

python -m textblob.download_corpora    {Necessary Lingustic Resources}

5. Prepare the Dataset

       Required Name for Code: 3-6-25_ai_dev_assignment_tickets_complex_1000.xls - OpTransactionHistoryUX3.csv
       Either change the file name in the code to whatever yours is, or use this name.
       The file needs to be .csv


Run Application

To process data, train classification models, then save as .pkl files in project directory.

    Output in the terminal will show the accuracy scores of the trained models and some example predictions. 
    This step will also create the .pkl files (e.g., tfidf_vectorizer.pkl, issue_type_classifier.pkl, etc.) in project folder.

The Gradio app needs te .pkl files to function.

Step 2: Launch the Gradio Web Interface

Once the models are trained and saved, launch the web app.

    Gradio will start a local web server. A local URL will be printed in the terminal (e.g., http://127.0.0.1:7860/).

    Open this URL in web browser.

Using the Gradio Interface

    Input: You will see a large text box labeled "Enter Customer Support Ticket Text Here". Type or paste any customer support ticket text into this box.

    Submit: Click the "Submit" button to run your machine learning pipeline on the entered text. The predicted issue type, urgency level, and extracted entities will appear in the output boxes on the right.

    Clear: Click the "Clear" button to clear the input and output fields.

    Flag: The "Flag" button is a built-in Gradio feature for feedback. If you find a prediction particularly good or bad, clicking "Flag" saves the input and output to a local CSV file (usually in a flagged subfolder). This data can be used later to improve the model. It does not re-run the model or change its behavior.

    Sophisticated Entity Extraction: Utilize dedicated Named Entity Recognition (NER) libraries (e.g., spaCy) for more robust and scalable entity extraction.

    Model Optimization: Experiment with different machine learning models (e.g., Random Forest, SVM, Gradient Boosting) and perform hyperparameter tuning using cross-validation to find the best performing models.

    Error Handling and Edge Cases: Improve error handling for unexpected inputs or data formats.
