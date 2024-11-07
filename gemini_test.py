import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel
import os

# Set page config
st.set_page_config(page_title="German Compound Word Analyzer", page_icon="📚")

# Initialize Vertex AI
def initialize_vertexai():
    try:
        project_id = os.getenv("germantextanalyzerllm")
        vertexai.init(project=project_id, location="us-central1")
        return True
    except Exception as e:
        st.error(f"Error initializing Vertex AI: {str(e)}")
        return False

def analyze_compounds(text):
    try:
        model = GenerativeModel(model_name="gemini-1.5-flash-002")
        prompt = f"""
        Analyze the following German text for compound words (Komposita).
        For each compound word found:
        1. Identify if it's a regular compound (Kompositum) or Konfixkompositum
        2. Break down its components
        3. Explain its formation and meaning in German
        4. Suggest simpler alternatives if applicable

        Text: {text}

        Format the response in German as:
        ### Gefundene Komposita:
        1. [Wort]
           - Typ: [Reguläres Kompositum/Konfixkompositum]
           - Komponenten: [Teil1 + Teil2 (+ Teil3 wenn vorhanden)]
           - Erklärung: [Bedeutung und Bildung]
           - Alternative: [Einfachere Alternative, falls möglich]
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error analyzing compounds: {str(e)}"

def main():
    st.title("📚 Deutscher Komposita-Analyzer")
    st.write("Analysiere zusammengesetzte Wörter in deutschen Texten")

    # Initialize Vertex AI
    if not initialize_vertexai():
        st.stop()

    # Create input area
    text_input = st.text_area(
        "Gib deinen Text ein:",
        height=150,
        placeholder="Beispiel: Die Bundestagsabgeordnete diskutierte über Klimaschutzmaßnahmen in der Arbeitsgemeinschaft."
    )

    # Add analyze button
    if st.button("Analysieren", type="primary"):
        if text_input:
            with st.spinner("Analysiere Text..."):
                # Display analysis
                analysis = analyze_compounds(text_input)
                st.markdown(analysis)
        else:
            st.warning("Bitte gib einen Text ein.")

if __name__ == "__main__":
    main()
