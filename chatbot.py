# --- Imports and Setup ---
import os
import pickle
import pandas as pd
import spacy
import sqlite3
import requests
import nltk 
import sys
from geopy.geocoders import Nominatim
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# NLTK Preprocessing Imports
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

# --- 1. LOAD PRE-TRAINED MODELS (Loaded ONCE at startup) ---
# print("Loading models...")
try:
    with open('weather_chatbot_classifier.pkl', 'rb') as f:
        CLASSIFIER = pickle.load(f)
    with open('weather_chatbot_vectorizer.pkl', 'rb') as f:
        VECTORIZER = pickle.load(f)
    NLP_MODEL = spacy.load('en_core_web_sm')
    GEOLOCATOR = Nominatim(user_agent="weather_chatbot_telegram")
    
    load_dotenv()
    USE_MOCK_DATA_STR = os.getenv('USE_MOCK_DATA', 'True')
    USE_MOCK_DATA = USE_MOCK_DATA_STR.lower() in ('true', '1', 't')
    
    print("Models loaded successfully.")
    print(f"** Current Mode: {'MOCK' if USE_MOCK_DATA else 'LIVE API'} **")
except FileNotFoundError as e:
    print(f"Error: Model file not found. {e}")
    sys.exit("Please run train.py first to create the model files.")

# --- 2. DEFINE RESPONSE TEMPLATES (Global) ---
responses_df = pd.DataFrame({
    'intent': [
        'ask_temperature', 'ask_weather_conditions', 'ask_wind_speed',
        'ask_rain', 'ask_snowfall', 'ask_snow_depth', 'greet', 'unknown_intent',
        'ask_capabilities', 'goodbye'
    ], 'response_template': [
        "The temperature {time_period} is {temperature}.",
        "The condition {time_period} is: {weather_conditions}.",
        "The wind speed {time_period} is {wind_speed}.",
        "The rainfall {time_period} is {rain}.",
        "The snowfall {time_period} is {snowfall}.",
        "The snow depth {time_period} is {snow_depth}.",
        "Hello! How can I help you today?",
        "I'm sorry, I don't understand that request.",
        "I can provide weather information like temperature, conditions, and wind speed for today and tomorrow.",
        "Goodbye! Have a great day."
    ]
})

# --- 3. PREPROCESSING FUNCTIONS (Required for classify_user_query) ---
def preprocess(sentence):
    """Converts sentence to lowercase and tokenizes it."""
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    return tokens

def extract_feature(text):
    """Applies preprocessing, POS tagging, and lemmatization."""
    lmtzr = WordNetLemmatizer()
    words = preprocess(text)
    tags = nltk.pos_tag(words)
    features = []
    for word, tag in tags:
        if tag.startswith(('NN', 'VB', 'RB', 'JJ', 'WP', 'PRP', 'MD')):
            pos = tag[0].lower() if tag[0] in 'NV' else 'n'
            features.append(lmtzr.lemmatize(word, pos=pos))
    return features

def word_feats(words):
    """Implements bag of words."""
    return dict([(word, True) for word in words])

# --- 4. CHATBOT HELPER FUNCTIONS ---
def classify_user_query(query, classifier, vectorizer):
    """Classifies a query and returns confident intents and the top time entity."""
    INTENT_THRESHOLD = 0.20
    features = extract_feature(query)
    query_features = word_feats(features)
    query_vectorized = vectorizer.transform([query_features])

    probabilities = classifier.predict_proba(query_vectorized)
    all_class_labels = classifier.classes_

    # Process Intents
    intent_labels, intent_probs = all_class_labels[0], probabilities[0][0]
    sorted_intents = sorted(zip(intent_labels, intent_probs), key=lambda item: item[1], reverse=True)
    confident_intents = [(intent, prob) for intent, prob in sorted_intents if prob >= INTENT_THRESHOLD]

    # Process Time
    time_labels, time_probs = all_class_labels[1], probabilities[1][0]
    sorted_times = sorted(zip(time_labels, time_probs), key=lambda item: item[1], reverse=True)
    top_one_time = sorted_times[:1]
    
    return confident_intents, top_one_time

def create_weather_database():
    """Creates the SQLite database and table."""
    conn = None
    try:
        conn = sqlite3.connect('weather_database.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS weather_info (
                city TEXT, time_entity TEXT, temperature TEXT, weather_conditions TEXT,
                wind_speed TEXT, rain TEXT, snowfall TEXT, snow_depth TEXT,
                PRIMARY KEY (city, time_entity)
            )
        ''')
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error creating database: {e}")
    finally:
        if conn: conn.close()

def get_weather_info(city, time_entity, use_mock, geolocator):
    """Fetches weather data for a specific city and time."""
    if time_entity in ['no_time', 'now']:
        time_entity = 'today'

    conn = None
    try:
        conn = sqlite3.connect('weather_database.db')
        cursor = conn.cursor()
        #print(f"--- Checking database for: {city} ({time_entity}) ---")
        
        cursor.execute("SELECT temperature, weather_conditions, wind_speed, rain, snowfall, snow_depth FROM weather_info WHERE city=? AND time_entity=?", (city, time_entity))
        cached_result = cursor.fetchone()

        if cached_result:
            #print("--> Found in cache.")
            return {'temperature': cached_result[0], 'weather_conditions': cached_result[1], 'wind_speed': cached_result[2], 'rain': cached_result[3], 'snowfall': cached_result[4], 'snow_depth': cached_result[5]}
        #print("--> Not in cache. Fetching new data...")
        
        weather_info = {}
        if use_mock:
            #print("--> Using MOCK data source.")
            if time_entity == 'today':
                weather_info = {'temperature': '25°C', 'weather_conditions': 'Sunny', 'wind_speed': '15 km/h', 'rain': '0 mm', 'snowfall': '0 mm', 'snow_depth': '0 cm'}
            elif time_entity == 'tomorrow':
                weather_info = {'temperature': '22°C', 'weather_conditions': 'Partly Cloudy', 'wind_speed': '20 km/h', 'rain': '2 mm', 'snowfall': '0 mm', 'snow_depth': '0 cm'}
        else:
            #print("--> Calling LIVE API...")
            try:
                location = geolocator.geocode(city)
                if not location: return {}
                lat, lon = location.latitude, location.longitude
                api_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,weather_code,wind_speed_10m_max,rain_sum&temperature_unit=celsius&wind_speed_unit=kmh&timezone=auto"
                response = requests.get(api_url)
                response.raise_for_status()
                data = response.json()['daily']
                units = response.json()['daily_units']
                day_index = 0 if time_entity == 'today' else 1
                weather_info = {
                    'temperature': f"{data.get('temperature_2m_max', ['N/A'])[day_index]}{units.get('temperature_2m_max', '')}",
                    'weather_conditions': f"{data.get('weather_code', ['N/A'])[day_index]}",
                    'wind_speed': f"{data.get('wind_speed_10m_max', ['N/A'])[day_index]}{units.get('wind_speed_10m_max', '')}",
                    'rain': f"{data.get('rain_sum', ['N/A'])[day_index]}{units.get('rain_sum', '')}",
                    'snowfall': 'N/A', 'snow_depth': 'N/A'
                }
            except (requests.RequestException, KeyError, IndexError) as e:
                print(f"API request or parsing error: {e}")
                return {}

        if weather_info:
            cursor.execute('''INSERT OR REPLACE INTO weather_info VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', (city, time_entity, weather_info['temperature'], weather_info['weather_conditions'], weather_info['wind_speed'], weather_info['rain'], weather_info['snowfall'], weather_info['snow_depth']))
            conn.commit()
            # print("--> New data saved to database.")
        return weather_info
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {}
    finally:
        if conn: conn.close()

def extract_cities(text, nlp_model):
    """Extracts city names from text using spaCy NER."""
    doc = nlp_model(text)
    return [entity.text for entity in doc.ents if entity.label_ == 'GPE']

def generate_responses(intents, weather_info, responses_df, time_entity):
    """Generates response strings that include the relevant time period."""
    response_list = []
    time_period_str = 'for today' if time_entity == 'today' else 'for tomorrow' if time_entity == 'tomorrow' else 'now'
    weather_info_with_time = weather_info.copy()
    weather_info_with_time['time_period'] = time_period_str

    for intent_tuple in intents:
        intent_label = intent_tuple[0]
        if intent_label == 'greet':
            response_list.append(responses_df[responses_df['intent'] == 'greet']['response_template'].values[0])
            continue
        template_series = responses_df[responses_df['intent'] == intent_label]['response_template']
        if not template_series.empty:
            response_list.append(template_series.values[0].format(**weather_info_with_time))
    return response_list

# --- 5. HELPER FUNCTION FOR REPLYING ---
async def fetch_and_respond(update: Update, context: ContextTypes.DEFAULT_TYPE, predicted_intents, time_entity, cities):
    """A helper function to fetch data and send the final reply."""
    
    final_responses_text = ""
    requires_city = any(intent[0] not in ['ask_capabilities'] for intent in predicted_intents)
    
    if not requires_city:
         responses = generate_responses(predicted_intents, {}, responses_df, time_entity)
         final_responses_text = "\n".join(f"- {res}" for res in responses)
    else:
        for city in cities:
            weather_info = get_weather_info(city, time_entity, use_mock=USE_MOCK_DATA, geolocator=GEOLOCATOR)
            if weather_info:
                responses = generate_responses(predicted_intents, weather_info, responses_df, time_entity)
                final_responses_text += f"For {city}:\n" + "\n".join(f"- {res}" for res in responses) + "\n\n"
            else:
                final_responses_text += f"Sorry, I couldn't find weather information for {city}.\n"

    # Send the final message
    if final_responses_text:
        await update.message.reply_text(final_responses_text.strip())
    else:
        await update.message.reply_text(responses_df[responses_df['intent'] == 'unknown_intent']['response_template'].values[0])

# --- 6. DEFINE TELEGRAM HANDLER FUNCTIONS  ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message and clears any previous state."""
    context.user_data.clear() 
    await update.message.reply_text('Hello! I am a weather bot. Ask me about the weather for a specific city.')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """The core function that processes every user message with state."""
    user_query = update.message.text
    expecting = context.user_data.get('expecting')

    if expecting == 'city':
        cities = extract_cities(user_query, NLP_MODEL)
        if not cities:
            await update.message.reply_text("I'm sorry, that doesn't look like a city. Please try again, or ask a new question.")
            return 
        predicted_intents = context.user_data.get('saved_intents')
        time_entity = context.user_data.get('saved_time')
        context.user_data.clear()
        await fetch_and_respond(update, context, predicted_intents, time_entity, cities)

    else:
        predicted_intents, predicted_time = classify_user_query(user_query, CLASSIFIER, VECTORIZER)
        time_entity = predicted_time[0][0] if predicted_time else 'no_time'
        cities = extract_cities(user_query, NLP_MODEL)
        intent_labels = [intent[0] for intent in predicted_intents]

        # Handle high-priority intents
        if 'goodbye' in intent_labels:
            await update.message.reply_text(responses_df[responses_df['intent'] == 'goodbye']['response_template'].values[0])
            context.user_data.clear() # Clear state on goodbye
            return

        if 'greet' in intent_labels and len(intent_labels) == 1:
            await update.message.reply_text(responses_df[responses_df['intent'] == 'greet']['response_template'].values[0])
            context.user_data.clear()
            return
            
        if not predicted_intents:
            await update.message.reply_text(responses_df[responses_df['intent'] == 'unknown_intent']['response_template'].values[0])
            context.user_data.clear()
            return

        # Handle missing information
        requires_city = any(intent[0] not in ['ask_capabilities'] for intent in predicted_intents)
        
        if requires_city and not cities:
            # --- SAVE THE STATE ---
            context.user_data['expecting'] = 'city'
            context.user_data['saved_intents'] = predicted_intents
            context.user_data['saved_time'] = time_entity
            
            await update.message.reply_text("I can help with that! For which city would you like the weather forecast?")
            return

        # If all info is present, fetch data and respond
        await fetch_and_respond(update, context, predicted_intents, time_entity, cities)

# --- 7. MAIN FUNCTION TO SET UP AND RUN THE BOT ---
def main() -> None:
    """Sets up the Telegram bot and starts it."""
    token = os.getenv('TOKEN')
    if not token:
        sys.exit("Error: TOKEN environment variable not found. Please create a .env file.")

    application = Application.builder().token(token).build()
    create_weather_database()

    # This is where the handlers are "registered"
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot is running and polling for messages...")
    application.run_polling()

if __name__ == '__main__':
    # Download NLTK data needed for preprocessing
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)

    # Start the bot
    main()