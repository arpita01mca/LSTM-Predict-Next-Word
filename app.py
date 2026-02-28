import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences   

## Load the LSTM model
model=load_model('next_word_prediction_model.h5')

## Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

## Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list=tokenizer.texts_to_sequences([text])[0]
    if len(token_list) > max_sequence_len:
        token_list=token_list[-(max_sequence_len-1):] ## Ensure the sequence mathes the model's expected input length
    token_list=pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted=model.predict(token_list, verbose=0)
    predicted_word_index=np.argmax(predicted,axis=1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# --- Streamlit UI ---
import streamlit as st

st.set_page_config(page_title="Next Word Predictor", page_icon="🤖", layout="centered")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Next Word Prediction 🤖</h1>", unsafe_allow_html=True)
st.markdown("Select an example or type your own sequence of words below, and the model will predict the next word.", unsafe_allow_html=True)

# --- Example phrases for dropdown ---
examples = [
    "to be or not to",
    "what a piece of",
    "the lady doth",
    "is this a dagger",
    "tomorrow and tomorrow and",
    "o, what a",
    "if music be the"
]

# Dropdown for examples
example_choice = st.selectbox("Try an example:", ["--Select an example--"] + examples)

# Input section: user can type or use dropdown selection
input_text = st.text_input("Or type your own text here:", value=example_choice if example_choice != "--Select an example--" else "to be or not to")

# Automatically get max sequence length from the model
max_sequence_len = model.input_shape[1] + 1

# Prediction button
if st.button("Predict Next Word"):
    if input_text.strip() == "":
        st.warning("Please enter some text to predict the next word.")
    else:
        # Predict next word
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        if next_word:
            st.success(f"Predicted next word: **{next_word}**")
        else:
            st.error("Could not predict the next word. Try a different input.")

# Optional: Show your input and the tokenized sequence for debugging
with st.expander("See tokenized input (optional)"):
    token_list = tokenizer.texts_to_sequences([input_text])[0]
    st.write(token_list)