# 🤖 ML-Powered Weather Chatbot (Telegram)

This project is a fully functional, conversational AI chatbot that understands user queries about the weather. It is built in Python and connects to the Telegram messaging platform.

Instead of relying on simple `if/else` rules, this bot uses a **Machine Learning model** (a Naive Bayes classifier) trained on example data to understand user **intents** (like `ask_temperature` or `greet`) and **entities** (like `today` or `tomorrow`). It also uses `spaCy` for Named Entity Recognition (NER) to extract city names.

## ✨ Key Features

* **NLU Model**: A custom-trained `MultiOutputClassifier` (using `scikit-learn`) to predict both user intent and the desired time entity.
* **Telegram Integration**: Runs 24/7 as a live Telegram bot using the `python-telegram-bot` library.
* **Multi-Intent Handling**: Can handle complex queries with multiple intents (e.g., "Hello, what's the temperature?").
* **Conversational Logic**: Politely handles greetings, goodbyes, and requests for its own capabilities.
* **Data Caching**: Caches weather data in an `SQLite` database to reduce API calls and improve response speed.
* **Live/Mock Data**: Can be switched between a live weather API and a mock data source using a simple configuration.

---

## 🏗️ Project Structure

This project is split into two main scripts, separating the model training from the live application:

1.  **`train.py`**
    * This is the "model factory."
    * It reads your dataset (from the code or a CSV).
    * It processes the text, extracts features, and trains the `MultiOutputClassifier` and `DictVectorizer`.
    * It saves the two trained components as `.pkl` files (`weather_chatbot_classifier.pkl` and `weather_chatbot_vectorizer.pkl`).
    * You only run this script **once** to create the model files, or again when you want to update your training data.

2.  **`chatbot.py`**
    * This is the "live bot" application.
    * It **loads** the pre-trained `.pkl` files on startup.
    * It connects to the Telegram API using your secret token.
    * It runs an `asyncio` loop to listen for user messages 24/7.
    * When a message arrives, it uses the loaded models to classify the text and generate a response.



---

## 🚀 How to Set Up and Deploy

Follow these steps to get the bot running.

### Step 1: Get Your Telegram Bot Token

Your bot needs a secret token to connect to Telegram's API.

1.  Open your Telegram app and search for the user **`@BotFather`**.
2.  Start a chat with `BotFather` and send the command `/newbot`.
3.  Follow the instructions to choose a name and a username for your bot.
4.  When you're done, `BotFather` will send you a message with your **API token**. It will look something like `1234567890:ABC-DEF1234ghIkl-zyx57W2v1u123456`.
5.  **Copy this token and keep it secret.**



### Step 2: Create Your `.env` File

Never paste your secret token directly into your code. You must use an environment file.

1.  In your project's root folder, create a new file named `.env`.
2.  Open the file in a text editor and add the following lines, pasting your token inside the quotes:

    ```ini
    TOKEN="YOUR_TELEGRAM_TOKEN_GOES_HERE"
    USE_MOCK_DATA="True"
    ```
    * `TOKEN`: This is the secret key for your bot.
    * `USE_MOCK_DATA`: Set this to `True` to use your mock data, or `False` to call the live weather API.

### Step 3: Create Your `.gitignore` File

To protect your secrets, you must tell Git to ignore your `.env` file.

1.  In your project folder, create a new file named `.gitignore`.
2.  Add the following lines. This will prevent you from ever accidentally uploading your secrets, database, or model files to GitHub.

    ```
    # Environment file with secrets
    .env

    # Python cache
    __pycache__/

    # Database file
    *.db

    # Saved model files
    *.pkl
    ```

### Step 4: Install Dependencies

All required Python libraries are listed in the requirements.txt file.

1. Open your terminal and navigate to your project folder.
2. Activate your virtual environment (if you are using one).
3. Install all required libraries with this single command:
    ```Bash
    pip install -r requirements.txt
    ```
4. You also need to download the spaCy language model:
    ```Bash
    python -m spacy download en_core_web_sm
    ```

### Step 5: Train Your Model (Run Once)

Before you can start the bot, you must create the `.pkl` model files.

1.  If your `train.py` script prompts for a CSV file, make sure it's ready.
2.  Run the training script from your terminal:

    ```Bash
    python train.py
    ```
3.  You should see output about the model's accuracy, and two new files (`weather_chatbot_classifier.pkl` and `weather_chatbot_vectorizer.pkl`) will appear in your folder.

### Step 6: Run Your Bot!

You are now ready to launch your chatbot.

1.  From your terminal, run the main bot script:
    ```sh
    python chatbot.py
    ```
2.  You should see the "Loading models..." and "Bot is running..." messages.
3.  Open Telegram, find your bot by its username, and send it a message!

### Step 7: Keep it Running (Deployment for Your Professor)

Your bot will stop as soon as you close the terminal. To keep it running 24/7 for your professor to test, you must host it on a server. The easiest way is with a free cloud service.

**Recommended Option: [Railway.app](https://railway.app) (GitHub-based)**

1.  **Push your project to GitHub** (make sure your `.gitignore` is working and `.env` is **not** uploaded).
2.  Sign up for a free account on Railway.
3.  Create a new project and link it to your GitHub repository.
4.  In your project's settings on Railway, go to the **"Variables"** tab.
5.  Add your secrets here. They must match the names in your code:
    * `TOKEN` = (your Telegram token)
    * `USE_MOCK_DATA` = `True`
6.  Railway will ask for a **"Start Command"**. Enter:
    ```Bash
    python chatbot.py
    ```
Railway will automatically read your `requirements.txt`, install everything, and run your bot. It will stay online 24/7.

**Option 2: PythonAnywhere (https://PythonAnywhere.com) (Manual Upload)**
1. Sign up for a free "Beginner" account.
2. Upload Files: Go to the "Files" tab. Upload your project files (chatbot.py, train.py, requirements.txt, .env, and your .pkl models) into a new directory.
3. Set Up Environment:
    * Open the "Consoles" tab and start a Bash console.
    * Navigate into your project folder: cd your-project-folder
    * Create a virtual environment: python -m venv venv
    * Activate it: source venv/bin/activate
    * Install libraries: pip install -r requirements.txt
    * Download NLTK/spaCy models: python -m spacy download en_core_web_sm, etc.
4. Set Up 24/7 Task:
    * Go to the "Tasks" tab.
    * Find "Always-on tasks" and click "Add new Always-on task".
    * Enter the full command to run your bot, including the full path to your virtual environment's Python and your script:
        ```Bash
        /home/YourUsername/your-project-folder/venv/bin/python /home/YourUsername/your-project-folder/chatbot.py
        ```
    **(Replace YourUsername and your-project-folder with your details)**

Your bot will now run 24/7. PythonAnywhere reads the secrets directly from the .env file you uploaded.