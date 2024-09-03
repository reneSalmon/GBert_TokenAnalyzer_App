import streamlit as st
from transformers import BertTokenizer

# Load the model's tokenizer
pretrained_weights = 'deepset/gbert-large'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

# Function definition for tokenizing and print output
def tokenize_text(text):
    # Tokenize the text
    token_list = tokenizer.tokenize(text)

    # Display tokenized sentences
    st.write("Tokenized Sentences:")
    sentence_list = []
    for token in token_list:
        sentence_list.append(token)
        if token == '.':
            st.write(sentence_list)
            sentence_list = []

    # Display the number of tokens
    st.write('Die Anzahl der Tokens ist:', len(token_list))

# Streamlit app layout
st.title('Text Tokenizer using BERT')
st.write("Please enter text to tokenize:")

# Text input for user to enter text (initially empty)
input_text = st.text_area('Enter your text here', value='')

# Tokenize and display output when button is clicked
if st.button('Tokenize Text'):
    if input_text.strip():  # Only tokenize if there is some input
        tokenize_text(input_text)
    else:
        st.write("Please enter some text to tokenize.")
