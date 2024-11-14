import streamlit as st
try:
    from transformers import AutoTokenizer
    from nltk.probability import FreqDist
    import vertexai
except ImportError as e:
    st.error(f"Required packages are missing. Please check requirements.txt")
    st.stop()

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

import math
import pandas as pd
import re
from typing import Dict, List
import os
from vertexai.generative_models import GenerativeModel

# Access secrets
credentials = st.secrets["gcp_service_account"]

@st.cache_resource
def initialize_gemini():
    try:
        vertexai.init(
            project="lewagon-batch672",
            location="us-central1"
        )
        model = GenerativeModel(
            model_name="gemini-1.5-flash-002",
            generation_config={
                "temperature": 0,
                "top_k": 1,
                "top_p": 0.1,
                "max_output_tokens": 1024
            }
        )
        return model
    except Exception as e:
        st.error(f"Failed to initialize Vertex AI: {str(e)}")
        return None

def calculate_shannon_entropy(token_list):
    freq_dist = FreqDist(token_list)
    total_words = len(token_list)
    probabilities = [freq_dist[word] / total_words for word in freq_dist]
    entropy = -sum([p * math.log2(p) for p in probabilities if p > 0])
    return entropy

def calculate_gunning_fog_index(text, token_list):
    sentences = text.count('.') + text.count('?') + text.count('!')
    if sentences == 0:
        sentences = 1
    complex_words = sum(1 for word in token_list if len(word) >= 3 and not word.istitle())
    fog_index = 0.4 * ((len(token_list) / sentences) + (100 * complex_words / len(token_list)))
    return fog_index

class TextQualityAnalyzer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-base")
        self.complex_word_threshold = 3
        self.gemini_model = initialize_gemini()

    def analyze_text(self, text: str) -> Dict:
        try:
            tokens = self.tokenizer.tokenize(text)
            token_count = len(tokens)
            entropy = calculate_shannon_entropy(tokens)
            fog_index = calculate_gunning_fog_index(text, tokens)
            return {
                "tokens": tokens,
                "token_count": token_count,
                "entropy": entropy,
                "fog_index": fog_index,
                "quality_checks": self.run_quality_checks(text, token_count, entropy, fog_index)
            }
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            return None

    def run_quality_checks(self, text: str, token_count: int, entropy: float, fog_index: float) -> List[Dict]:
        return [
            self.check_text_length(token_count),
            self.check_gunning_fog(fog_index),
            self.check_shannon_entropy(entropy),
            self.check_gender_language(text),
            self.check_anglicisms(text),
            self.check_sentence_length(text),
            self.analyze_compounds_with_gemini(text)
        ]

    def check_text_length(self, token_count: int) -> Dict:
        is_valid = 50 <= token_count <= 256
        optimal = 60 <= token_count <= 140
        return {
            "name": "Textlänge",
            "passed": is_valid,
            "message": f"Token count: {token_count} ({'optimal' if optimal else 'outside optimal range'})",
            "requirement": "Token size should be between 50-256 (optimal: 60-140)"
        }

    def check_gunning_fog(self, fog_index: float) -> Dict:
        is_valid = 8 <= fog_index <= 18
        optimal = 9 <= fog_index <= 15
        return {
            "name": "Textkomplexität",
            "passed": is_valid,
            "message": f"Gunning-Fog Index: {fog_index:.2f}",
            "requirement": "Gunning-Fog Index should be between 8-18 (optimal: 9-15)"
        }

    def check_shannon_entropy(self, entropy: float) -> Dict:
        is_valid = 4.5 <= entropy <= 7.0
        optimal = 5.0 <= entropy <= 6.5
        return {
            "name": "Informationsdichte",
            "passed": is_valid,
            "message": f"Shannon Entropy: {entropy:.2f}",
            "requirement": "Shannon Entropy should be between 4.5-7.0 (optimal: 5.0-6.5)"
        }

    def check_gender_language(self, text: str) -> Dict:
        gender_patterns = r'(\w+:innen|\w+\/innen|\w+\*innen)'
        matches = re.findall(gender_patterns, text)
        return {
            "name": "Gendern",
            "passed": len(matches) == 0,
            "message": f"Found gendered terms: {', '.join(matches) if matches else 'None'}",
            "requirement": "Avoid gendered language forms"
        }

    def check_anglicisms(self, text: str) -> Dict:
        try:
            prompt = """
            Analyze the following German text for anglicisms (English words or phrases used in German).
            For each anglicism found:
            1. Identify the anglicism
            2. Explain why it's an anglicism
            3. Suggest a German alternative
            4. Enter Paragraph before listing next anglicism

            Text: {text}

            Format the response in German as:
            ### Gefundene Anglizismen:
            1. [Anglizismus]
               - Erklärung: [Warum es ein Anglizismus ist]
               - Alternative: [Deutsche Alternative]

            If no anglicisms are found, respond with: "Keine Anglizismen gefunden"
            """
            response = self.gemini_model.generate_content(prompt)
            return {
                "name": "Anglizismen",
                "passed": "Keine Anglizismen gefunden" in response.text,
                "message": response.text,
                "requirement": "Avoid unnecessary anglicisms"
            }
        except Exception as e:
            return {
                "name": "Anglizismen",
                "passed": True,
                "message": f"Error in analysis: {str(e)}",
                "requirement": "Avoid unnecessary anglicisms"
            }

    def check_sentence_length(self, text: str) -> Dict:
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        long_sentences = [s for s in sentences if len(s.split()) > 20]
        return {
            "name": "Satzlänge",
            "passed": len(long_sentences) == 0,
            "message": f"Found {len(long_sentences)} sentences longer than 20 words",
            "requirement": "Sentences should not exceed 20 words"
        }

    def display_tokenized_sentences(self, tokens: List[str]) -> List[List[str]]:
        sentences = []
        current_sentence = []
        for token in tokens:
            current_sentence.append(token)
            if token == '.':
                sentences.append(current_sentence)
                current_sentence = []
        if current_sentence:
            sentences.append(current_sentence)
        return sentences

    def analyze_compounds_with_gemini(self, text: str) -> Dict:
        try:
            prompt = """
            Analyze the following German text for compound words (Komposita).
            For each compound word found:
            1. Identify if it's a regular compound (Kompositum) or Konfixkompositum
            2. Break down its components
            3. Explain its formation and meaning in German
            4. Suggest simpler alternatives if applicable
            5. Enter Paragraph before listing next result

            Text: {text}

            Format the response in German as:
            ### Gefundene Komposita:
            1. [Wort + (Reguläres Kompositum/Konfixkompositum)]
               - Komponenten: [Teil1 + Teil2 (+ Teil3 wenn vorhanden)]
               - Alternative Formulierung: [Einfachere Alternative, falls möglich oder Komponenten voll auschreiben und mit Bindesstrich verbinden]

            If no compounds are found, respond with: "Keine Komposita gefunden"
            """
            response = self.gemini_model.generate_content(prompt)
            return {
                "has_compounds": "Keine Komposita gefunden" not in response.text,
                "analysis": response.text
            }
        except Exception as e:
            return {
                "has_compounds": False,
                "analysis": f"Error analyzing compounds: {str(e)}"
            }

def display_results(analyzer: TextQualityAnalyzer, analysis: Dict):
    st.subheader("Tokenized Sentences")
    sentences = analyzer.display_tokenized_sentences(analysis['tokens'])
    for i, sentence in enumerate(sentences, 1):
        with st.expander(f"Sentence {i} ({len(sentence)} tokens)"):
            st.write(f"**Tokens:** {' '.join(sentence)}")
            st.write(f"**Word count:** {len([t for t in sentence if not t.startswith('##')])}")

    st.subheader("Quality Analysis Results")
    for check in analysis['quality_checks']:
        with st.expander(f"{check['name']} - {'✅' if check['passed'] else '❌'}"):
            st.write(f"**Requirement:** {check['requirement']}")
            st.write(f"**Result:** {check['message']}")

def main():
    st.title('GBert Text Quality Analyzer')

    try:
        analyzer = TextQualityAnalyzer()
        input_text = st.text_area('Enter your text here:', height=200, key="input_text_area")

        if st.button('Analyze Text', key="analyze_button"):
            if not input_text.strip():
                st.warning("Please enter some text to analyze.")
                return

            with st.spinner('Analyzing text...'):
                analysis = analyzer.analyze_text(input_text)
                if analysis:
                    display_results(analyzer, analysis)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
