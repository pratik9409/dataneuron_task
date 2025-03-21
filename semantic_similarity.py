import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Preprocess text
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Clean and preprocess the input text."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", " ", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Load the transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Define similarity functions with error handling
def transformer_similarity(text1, text2):
    try:
        embeddings1 = model.encode(text1, convert_to_tensor=True)
        embeddings2 = model.encode(text2, convert_to_tensor=True)
        return util.pytorch_cos_sim(embeddings1, embeddings2).item()
    except Exception as e:
        print(f"Transformer similarity failed: {e}")
        return 0.0

def tfidf_similarity(text1, text2):
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    except Exception as e:
        print(f"TF-IDF similarity failed: {e}")
        return 0.0

# Introduce weighted scoring
def compute_similarity(text1, text2):
    """Compute a weighted average similarity score from Transformer and TF-IDF."""
    text1, text2 = preprocess_text(text1), preprocess_text(text2)
    if not text1 or not text2:
        return 0.0
    
    sim1 = transformer_similarity(text1, text2)
    sim2 = tfidf_similarity(text1, text2)
    
    # 70% Transformer, 30% TF-IDF weight
    weighted_score = round((0.7 * sim1 + 0.3 * sim2), 2)
    return weighted_score

# Process the CSV file with error handling
def process_csv(input_file, output_file):
    try:
        data = pd.read_csv(input_file)
        if 'text1' not in data.columns or 'text2' not in data.columns:
            raise ValueError("CSV must contain 'text1' and 'text2' columns")
        
        data['similarity_score'] = data.apply(lambda row: compute_similarity(row['text1'], row['text2']), axis=1)
        data.to_csv(output_file, index=False)
        print(f"✅ Saved output to {output_file}")
    except Exception as e:
        print(f"Error processing CSV: {e}")

# Test with a sample CSV
if __name__ == "__main__":
    input_file = 'DataNeuron_Text_Similarity.csv'
    output_file = 'DataNeuron_Text_Similarity_Scored.csv'
    process_csv(input_file, output_file)
    
    # Test with a quick example
    text1 = "The quick brown fox jumps over the lazy dog"
    text2 = "A fast brown fox leaps over a sleeping dog"
    print(f"Sample Similarity Score: {compute_similarity(text1, text2)}")
