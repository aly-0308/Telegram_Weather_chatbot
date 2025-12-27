# --- Imports and Setup ---
import nltk
import pandas as pd
import pickle
import time
import sys  
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# --- 1. Getting the training and testing data Functions 
def read_csv_file(file_name):
    """Reads a CSV file and handles common errors."""
    try:
        df = pd.read_csv(file_name)
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file_name}' is empty.")
        return None
    except pd.errors.ParserError:
        print(f"Error: Unable to parse file '{file_name}'. Make sure it's a valid CSV.")
        return None

def check_dataframe(df):
    """Checks if a DataFrame has the required columns."""
    required_columns = ['query', 'intent', 'time_entity']
    return all(col in df.columns for col in required_columns)

def training_testing_df_preprocessing(df):
    """Cleans the training DataFrame."""
    print("Starting data preprocessing...")
    original_rows = len(df)
    
    # Drop rows where 'query' is NaN (empty)
    df.dropna(subset=['query'], inplace=True)
    # Keep only the rows where the 'query' is a string
    # This also effectively handles incorrectly typed data
    is_string_mask = df['query'].apply(lambda x: isinstance(x, str))
    df = df[is_string_mask]
    # Remove queries that are just empty whitespace
    df = df[df['query'].str.strip() != ''] 
    
    rows_removed = original_rows - len(df)
    if rows_removed > 0:
        print(f"Preprocessing complete. Removed {rows_removed} invalid rows.")
    else:
        print("Preprocessing complete. No invalid rows found.")
    return df

def get_train_test_data():
    """Handles the Deployer prompts to load and validate the training data CSV."""
    
    print('=' * 51 + "Chatbot Description" + '=' * 51)
    print("This script will train the NLU model for the weather chatbot.")
    print('-' * 125)
    print('=' * 51 + ' Chatbot Data Loader ' + '=' * 51)
    
    if input("Is the Training and Testing dataset file ready and in the same directory? (y/n): ").lower() != 'y':
        sys.exit("Script terminated: Please place the CSV file in the current directory and restart.")

    while True:
        try:
            file_name = input("Enter the name of the training dataset CSV file (e.g., chatbot_training_testing_data): ")
            full_path = file_name + '.csv'
            print('*' * 125)
            
            training_testing_df = read_csv_file(full_path)
    
            if training_testing_df is None:
                print('Unable to read the training dataset file. Please try again.')
                continue

            print('The training dataset was successfully read!')
            print('*' * 125)
            
            if check_dataframe(training_testing_df):
                print("The dataset meets the structural requirements (query, intent, time_entity).")
                training_testing_df = training_testing_df_preprocessing(training_testing_df)
            else:
                print("Error: The dataset does not have the required columns: 'query', 'intent', 'time_entity'.")
                continue
    
            # Print info about the file structure.
            print("\nThe dataset has three columns:")
            print("1. 'query' (string): The user's question.")
            print("2. 'intent' (string): The category of the question.")
            print("3. 'time_entity' (string): The time reference in the question.")
            print('*' * 125)
            
            print("Unique values for 'intent':")
            print(training_testing_df['intent'].unique())
            print('*' * 125)
    
            print("Unique values for 'time_entity':")
            print(training_testing_df['time_entity'].unique())
            print('*' * 125)
            
            print(training_testing_df.head())
            print('*' * 125)
            
            return training_testing_df # Success, return the DataFrame
            
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            print('*' * 125)

# --- 2. PREPROCESSING AND FEATURE EXTRACTION FUNCTIONS ---
def preprocess(sentence):
    """Converts sentence to lowercase, tokenizes, and removes stopwords."""
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    # filtered_words = [w for w in tokens if not w in stopwords.words('english')]
    return tokens

def extract_feature(text):
    """Applies preprocessing, POS tagging, and lemmatization."""
    lmtzr = WordNetLemmatizer()
    words = preprocess(text)
    tags = nltk.pos_tag(words)
    
    features = []
    for word, tag in tags:
        # MODIFIED: Added 'WP' (Wh-pronoun), 'PRP' (Personal pronoun),
        # and 'MD' (Modal verb) to the list of accepted tags.
        if tag.startswith(('NN', 'VB', 'RB', 'JJ', 'WP', 'PRP', 'MD')):
            pos = tag[0].lower() if tag[0] in 'NV' else 'n'
            features.append(lmtzr.lemmatize(word, pos=pos))
    return features

def word_feats(words):
    """Implements bag of words."""
    return dict([(word, True) for word in words])

def extract_features_from_doc(data):
    """Parses the whole document to create a feature set and separate label sets."""
    corpus = []
    intents = []
    times = []
    for index, row in data.iterrows():
        features = extract_feature(row['query'])
        corpus.append(word_feats(features))
        intents.append(row['intent'])
        times.append(row['time_entity'])
    return corpus, intents, times

# --- 3. MODEL TRAINING FUNCTION (NLTK Features + Logistic Regression) ---
def train_model(training_testing_df):
    """
    Trains the MultiOutputClassifier using NLTK-based features
    and a Logistic Regression model.
    """
    print('=' * 51 + ' Starting model training... ' + '=' * 51)
    model_training_start_time = time.time()
    
    # 1. Use your NLTK pipeline to extract features
    corpus, intents, times = extract_features_from_doc(training_testing_df)

    # 2. Use DictVectorizer (since we have dictionaries of features)
    vectorizer = DictVectorizer()
    X = vectorizer.fit_transform(corpus)
    
    # 3. Create the target DataFrame
    y = pd.DataFrame({'intent': intents, 'time_entity': times})

    # 4. Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Use LogisticRegression (the more powerful model)
    classifier = MultiOutputClassifier(LogisticRegression(max_iter=1000))
    classifier.fit(X_train, y_train)

    # 6. Evaluate the model
    predictions = classifier.predict(X_test)
    intent_accuracy = accuracy_score(y_test['intent'], predictions[:, 0])
    time_accuracy = accuracy_score(y_test['time_entity'], predictions[:, 1]) 
    
    print(f"Model Accuracy for Intents: {intent_accuracy * 100:.2f}%")
    print(f"Model Accuracy for Time: {time_accuracy * 100:.2f}%")
    print("Model training complete.")
    
    model_training_end_time = time.time()
    print(f"Model Training finished in {model_training_end_time - model_training_start_time:.2f} seconds.")
    print('*' * 125)
    
    # 7. Save the components
    print("Saving model components to disk...")
    with open('weather_chatbot_classifier.pkl', 'wb') as f:
        pickle.dump(classifier, f)
    with open('weather_chatbot_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    print("Training and saving complete.")
    print('-' * 125)

# --- 4. SCRIPT ENTRY POINT ---
if __name__ == "__main__":
    
    # Download NLTK data needed for preprocessing
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)

    # 1. Load and validate the data
    training_df = get_train_test_data()
    
    # 2. If data is loaded successfully, train the model
    if training_df is not None:
        train_model(training_df)
    else:
        print("Exiting script. Model was not trained because data was not loaded.")