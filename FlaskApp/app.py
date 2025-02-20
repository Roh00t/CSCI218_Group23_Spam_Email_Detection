from flask import Flask, request, render_template
import pickle
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
sys.modules["keras.preprocessing.text"] = tf.keras.preprocessing.text
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# --------------------------------------------------
# Define and Register Custom LSTM Class (if needed)
# --------------------------------------------------
@tf.keras.utils.register_keras_serializable()
class CustomLSTM(tf.keras.layers.LSTM):
    @classmethod
    def from_config(cls, config):
        config.pop("time_major", None)  # Remove if present
        return super().from_config(config)

# --------------------------------------------------
# Load Pretrained Models and Preprocessors
# --------------------------------------------------
# 1. Naïve Bayes Model (loaded via pickle)
naive_bayes_model = pickle.load(open("naive_bayes.pkl", "rb"))

# 2. LSTM Model (saved as HDF5; load using tf.keras.models.load_model)
lstm_model = tf.keras.models.load_model("lstm_model.h5", custom_objects={'LSTM': CustomLSTM, 'CustomLSTM': CustomLSTM})
lstm_tokenizer = pickle.load(open("lstm_tokenizer.pkl", "rb"))

# 3. Random Forest Model (loaded via pickle)
rf_model = pickle.load(open("random_forest.pkl", "rb"))

# 4. TF–IDF Vectorizers:
#    - For Naïve Bayes
vectorizer_nb = pickle.load(open("vectorizer.pkl", "rb"))
#    - For Random Forest (this vectorizer was saved separately)
rf_vectorizer = pickle.load(open("rf_vectorizer.pkl", "rb"))

# --------------------------------------------------
# Utility Functions
# --------------------------------------------------
def preprocess_text(text: str) -> str:
    """
    Basic text preprocessing: convert to lowercase and strip whitespace.
    Adjust as needed to match your training pipeline.
    """
    return text.lower().strip()

def predict_spam(email_text: str, model_choice: str):
    """
    Given an email text and a model choice ("naive_bayes", "lstm", or "random_forest"),
    returns "Spam" or "Not Spam" along with the spam probability.
    """
    # Preprocess input text
    email_text = preprocess_text(email_text)
    if not email_text:
        return {"label": "Please enter an email.", "probability": 0}
    
    if model_choice == "naive_bayes":
        # Use the TF-IDF vectorizer for Naïve Bayes.
        transformed_text = vectorizer_nb.transform([email_text])
        prob = naive_bayes_model.predict_proba(transformed_text)[0][1]  # Probability of Spam
    
    elif model_choice == "lstm":
        # For the LSTM, use the tokenizer and pad the sequences.
        sequences = lstm_tokenizer.texts_to_sequences([email_text])
        # Pad sequences to the length used during training (e.g., 100)
        padded_seq = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100)
        prob = float(lstm_model.predict(padded_seq)[0][0])  # Probability of Spam
    
    elif model_choice == "random_forest":
        # Transform using the Random Forest vectorizer.
        transformed_text = rf_vectorizer.transform([email_text])
        prob = rf_model.predict_proba(transformed_text)[0][1]  # Probability of Spam
    
    else:
        return {"label": "Invalid model choice.", "probability": 0}

    # Use threshold = 0.6 (adjust as needed)
    prediction_label = "Spam" if prob >= 0.6 else "Not Spam"
    return {"label": prediction_label, "probability": round(prob * 100, 2)}  # return dictionary

# --------------------------------------------------
# Flask Routes
# --------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    predictions = {}
    model_keys = {
        "Naïve Bayes": "naive_bayes",
        "LSTM": "lstm",
        "Random Forest": "random_forest"
    }

    total_probability = 0
    model_count = 0

    if request.method == "POST":
        email_text = request.form.get("email_text", "").strip()

        if email_text:  # Execute only if input exists
            for display_name, model_key in model_keys.items():
                prediction = predict_spam(email_text, model_key)
                predictions[display_name] = prediction  # Store prediction

                # Count only valid probability values
                if prediction["probability"] > 0:  
                    total_probability += prediction["probability"]
                    model_count += 1  

    # Compute average probability (handle cases where all probabilities are 0%)
    average_probability = (total_probability / model_count) if model_count > 0 else 0

    return render_template("index.html", predictions=predictions, average_probability=average_probability, model_count=model_count)

# --------------------------------------------------
# Main Entry
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)