import streamlit as st
import nltk
nltk.download('punkt')
from transformers import BertTokenizer
from nltk.probability import FreqDist
import math
import pandas as pd
import re
from typing import Dict, List


# Load the model's tokenizer
pretrained_weights = 'deepset/gbert-large'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

def calculate_shannon_entropy(token_list):
    """Calculates the Shannon entropy of a given text."""
    freq_dist = FreqDist(token_list)
    total_words = len(token_list)
    probabilities = [freq_dist[word] / total_words for word in freq_dist]
    entropy = -sum([p * math.log2(p) for p in probabilities if p > 0])
    return entropy

def calculate_gunning_fog_index(text, token_list):
    """Calculates the Gunning-Fog Index of a given text."""
    sentences = text.count('.') + text.count('?') + text.count('!')
    if sentences == 0:
        sentences = 1

    complex_words = sum(1 for word in token_list if len(word) >= 3 and not word.istitle())
    fog_index = 0.4 * ((len(token_list) / sentences) + (100 * complex_words / len(token_list)))
    return fog_index

class TextQualityAnalyzer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.anglicisms = ['cool', 'nice', 'fancy', 'update', 'queen', 'hollywood']
        self.complex_word_threshold = 3
        self.compound_markers = {
            'linking_elements': ['s', 'es', 'n', 'en', 'er', 'e'],
            'common_heads': ['stelle', 'haus', 'zeit', 'raum', 'mann', 'frau', 'kind'],
            'common_modifiers': ['haupt', 'grund', 'zeit', 'hand', 'land']
        }
        self.konfixes = ['bio', 'geo', 'phil', 'tele', 'therm', 'graph', 'phon',
                        'log', 'path', 'psych', 'trag', 'kom', 'techno']

    def analyze_text(self, text: str) -> Dict:
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

    def run_quality_checks(self, text: str, token_count: int, entropy: float, fog_index: float) -> List[Dict]:
        return [
            self.check_text_length(token_count),
            self.check_gunning_fog(fog_index),
            self.check_shannon_entropy(entropy),
            self.check_gender_language(text),
            self.check_anglicisms(text),
            self.check_sentence_length(text),
            self.check_compounds(text)
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
        found = [word for word in self.anglicisms if word.lower() in text.lower()]
        return {
            "name": "Anglizismen",
            "passed": len(found) == 0,
            "message": f"Found anglicisms: {', '.join(found) if found else 'None'}",
            "requirement": "Avoid unnecessary anglicisms"
        }

    def check_sentence_length(self, text: str) -> Dict:
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        long_sentences = [s for s in sentences if len(s.split()) > 20]
        return {
            "name": "Satzlänge",
            "passed": len(long_sentences) == 0,
            "message": f"Found {len(long_sentences)} long sentences: {'; '.join(long_sentences) if long_sentences else 'None'}",
            "requirement": "Sentences should not exceed 20 words"
        }

    def check_compounds(self, text: str) -> Dict:
        words = [w for w in text.split() if len(w) > 5]
        potential_compounds = []

        for word in words:
            tokens = self.tokenizer.tokenize(word)
            if len(tokens) > 1 and not any(token.startswith('##') for token in tokens):
                compound_info = {
                    'word': word,
                    'tokens': tokens,
                    'type': self._determine_compound_type(word, tokens)
                }
                potential_compounds.append(compound_info)

        return {
            "name": "Zusammengesetzte Wörter",
            "passed": True,
            "message": f"Found compounds: {', '.join(c['word'] for c in potential_compounds) if potential_compounds else 'None'}",
            "compounds": potential_compounds,
            "requirement": "Analysis of German compound words"
        }

    def _determine_compound_type(self, word: str, tokens: List[str]) -> str:
        konfix_count = sum(1 for konfix in self.konfixes if konfix in word.lower())
        if konfix_count >= 2:
            return "Konfixkompositum"
        elif konfix_count == 1:
            return "Konfix-Compound"
        else:
            return "Regular Compound"

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

import streamlit as st
from transformers import BertTokenizer
from nltk.probability import FreqDist
import math
import pandas as pd
import re
from typing import Dict, List

# Load the model's tokenizer
pretrained_weights = 'deepset/gbert-large'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

# Add NLTK download
import nltk
nltk.download('punkt')

def calculate_shannon_entropy(token_list):
    """Calculates the Shannon entropy of a given text."""
    freq_dist = FreqDist(token_list)
    total_words = len(token_list)
    probabilities = [freq_dist[word] / total_words for word in freq_dist]
    entropy = -sum([p * math.log2(p) for p in probabilities if p > 0])
    return entropy

def calculate_gunning_fog_index(text, token_list):
    """Calculates the Gunning-Fog Index of a given text."""
    sentences = text.count('.') + text.count('?') + text.count('!')
    if sentences == 0:
        sentences = 1

    complex_words = sum(1 for word in token_list if len(word) >= 3 and not word.istitle())
    fog_index = 0.4 * ((len(token_list) / sentences) + (100 * complex_words / len(token_list)))
    return fog_index

class TextQualityAnalyzer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.anglicisms = ['cool', 'nice', 'fancy', 'update', 'queen', 'hollywood']
        self.complex_word_threshold = 3
        self.compound_markers = {
            'linking_elements': ['s', 'es', 'n', 'en', 'er', 'e'],
            'common_heads': ['stelle', 'haus', 'zeit', 'raum', 'mann', 'frau', 'kind'],
            'common_modifiers': ['haupt', 'grund', 'zeit', 'hand', 'land']
        }
        self.konfixes = ['bio', 'geo', 'phil', 'tele', 'therm', 'graph', 'phon',
                        'log', 'path', 'psych', 'trag', 'kom', 'techno']

    def analyze_text(self, text: str) -> Dict:
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

    def run_quality_checks(self, text: str, token_count: int, entropy: float, fog_index: float) -> List[Dict]:
        return [
            self.check_text_length(token_count),
            self.check_gunning_fog(fog_index),
            self.check_shannon_entropy(entropy),
            self.check_gender_language(text),
            self.check_anglicisms(text),
            self.check_sentence_length(text),
            self.check_compounds(text)  # Added compound check
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
        found = [word for word in self.anglicisms if word.lower() in text.lower()]
        return {
            "name": "Anglizismen",
            "passed": len(found) == 0,
            "message": f"Found anglicisms: {', '.join(found) if found else 'None'}",
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

    def check_compounds(self, text: str) -> Dict:
        words = [w for w in text.split() if len(w) > 5]
        potential_compounds = []

        for word in words:
            tokens = self.tokenizer.tokenize(word)
            if len(tokens) > 1 and not any(token.startswith('##') for token in tokens):
                compound_info = {
                    'word': word,
                    'tokens': tokens,
                    'type': self._determine_compound_type(word, tokens)
                }
                potential_compounds.append(compound_info)

        return {
            "name": "Zusammengesetzte Wörter",
            "passed": True,
            "message": f"Found compounds: {', '.join(c['word'] for c in potential_compounds) if potential_compounds else 'None'}",
            "compounds": potential_compounds,
            "requirement": "Analysis of German compound words"
        }

    def _determine_compound_type(self, word: str, tokens: List[str]) -> str:
        konfix_count = sum(1 for konfix in self.konfixes if konfix in word.lower())
        if konfix_count >= 2:
            return "Konfixkompositum"
        elif konfix_count == 1:
            return "Konfix-Compound"
        else:
            return "Regular Compound"

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

def main():
    st.title('GBert Text Quality Analyzer')
    analyzer = TextQualityAnalyzer(tokenizer)

    input_text = st.text_area('Enter your text here:', height=200, key="input_text_area")

    if st.button('Analyze Text', key="analyze_button"):
        if input_text.strip():
            analysis = analyzer.analyze_text(input_text)

            # Display tokenized sentences
            st.subheader("Tokenized Sentences")
            sentences = analyzer.display_tokenized_sentences(analysis['tokens'])
            for i, sentence in enumerate(sentences, 1):
                with st.expander(f"Sentence {i} ({len(sentence)} tokens)"):
                    st.write(' '.join(sentence))

            # Display quality checks
            st.subheader("Quality Analysis Results")
            for check in analysis['quality_checks']:
                with st.expander(f"{check['name']} - {'✅' if check['passed'] else '❌'}"):
                    st.write(f"**Requirement:** {check['requirement']}")
                    st.write(f"**Result:** {check['message']}")
                    if 'compounds' in check:
                        for compound in check['compounds']:
                            st.write(f"- {compound['word']} ({compound['type']})")
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main()
