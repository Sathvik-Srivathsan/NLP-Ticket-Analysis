# NLP-Ticket-Analysis
Customer Support Ticket Analysis and Classification

This project develops a machine learning pipeline to classify customer support tickets by their issue type and urgency level, and to extract key entities such as product names, dates, and complaint keywords. It also provides an interactive web interface using Gradio for easy interaction.
📁 Project Structure

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

🚀 Setup Instructions

Follow these steps to set up and run the project on your local machine.
1. Prerequisites

    Python: Ensure you have Python 3.7 or newer installed. You can download it from python.org.

    Terminal/Command Prompt: You will need to use your system's terminal or command prompt to run commands.

2. Clone/Download the Project

    If you have a Git repository, clone it. Otherwise, download all the project files (Ticket.py, app.py, and the CSV file) into a single folder on your computer.

        Example Folder Path: C:\Users\YourUser\PycharmProjects\NLP Ticket Project\

3. Install Required Python Libraries

Open your terminal or command prompt, navigate to your project folder (using cd command), and run the following command:

cd "C:\Users\Sathvik Srivathsan\PycharmProjects\NLP Ticket Project\" # Replace with your actual path
pip install pandas scikit-learn nltk textblob joblib gradio

    pandas: For data manipulation.

    scikit-learn: For machine learning models and text feature extraction.

    nltk: Natural Language Toolkit for text preprocessing.

    textblob: For sentiment analysis and NLTK data download.

    joblib: For saving and loading models.

    gradio: For creating the web interface.

4. Download NLTK Data

NLTK requires additional data for text processing (like stopwords and lemmatization data). Run this command in your project folder:

python -m textblob.download_corpora

This will download necessary linguistic resources.
5. Prepare the Dataset

The project expects the dataset to be a CSV file with a specific name.

    Original File: 3-6-25_ai_dev_assignment_tickets_complex_1000.xls (your Excel file).

    Required Name for Code: 3-6-25_ai_dev_assignment_tickets_complex_1000.xls - OpTransactionHistoryUX3.csv

Action:

    Open your Excel file (3-6-25_ai_dev_assignment_tickets_complex_1000.xls) in Microsoft Excel or a compatible spreadsheet program.

    Go to File -> Save As.

    In the "Save As" dialog, change the "Save as type" to "CSV (Comma delimited) (*.csv)".

    Crucially, rename the file to exactly 3-6-25_ai_dev_assignment_tickets_complex_1000.xls - OpTransactionHistoryUX3.csv.

    Save this new CSV file in the same project folder where your Python scripts (Ticket.py and app.py) are located.

🏃 How to Run the Application

The application runs in two main steps: first, training the models, and then launching the web interface.
Step 1: Train the Machine Learning Models

This step will process your data, train the classification models, and save them as .pkl files in your project directory. These .pkl files are essential for the Gradio app to function.

    Open your terminal or command prompt.

    Navigate to your project folder.

    Run the Ticket.py script:

    python Ticket.py

    You will see output in the terminal showing the accuracy scores of the trained models and some example predictions. This step will also create the .pkl files (e.g., tfidf_vectorizer.pkl, issue_type_classifier.pkl, etc.) in your project folder.

Step 2: Launch the Gradio Web Interface

Once the models are trained and saved, you can launch the interactive web application.

    In the same terminal or command prompt (or a new one, still navigated to your project folder).

    Run the app.py script:

    python app.py

    Gradio will start a local web server. You will see a local URL printed in your terminal (e.g., http://127.0.0.1:7860/ or similar).

    Open this URL in your web browser.

Using the Gradio Interface

    Input: You will see a large text box labeled "Enter Customer Support Ticket Text Here". Type or paste any customer support ticket text into this box.

    Submit: Click the "Submit" button to run your machine learning pipeline on the entered text. The predicted issue type, urgency level, and extracted entities will appear in the output boxes on the right.

    Clear: Click the "Clear" button to clear the input and output fields.

    Flag: The "Flag" button is a built-in Gradio feature for feedback. If you find a prediction particularly good or bad, clicking "Flag" saves the input and output to a local CSV file (usually in a flagged subfolder). This data can be used later to improve the model. It does not re-run the model or change its behavior.

🛠️ Potential Improvements

While this project provides a solid foundation, here are some areas for future enhancements:

    Advanced Text Preprocessing: Implement contraction expansion (e.g., "don't" to "do not") and basic spelling correction for cleaner input.

    Richer Feature Engineering: Explore N-grams (combinations of words like "not working") or more advanced techniques like Word Embeddings (Word2Vec, BERT) for capturing deeper semantic meaning.

    Sophisticated Entity Extraction: Utilize dedicated Named Entity Recognition (NER) libraries (e.g., spaCy) for more robust and scalable entity extraction.

    Model Optimization: Experiment with different machine learning models (e.g., Random Forest, SVM, Gradient Boosting) and perform hyperparameter tuning using cross-validation to find the best performing models.

    Error Handling and Edge Cases: Improve error handling for unexpected inputs or data formats.
