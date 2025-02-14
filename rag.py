import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
import spacy
import pandas as pd
from scipy.spatial.distance import cosine
import numpy as np
import google.generativeai as genai

# Configure the Gemini API
GOOGLE_API_KEY = "Your_API_KEY"  
genai.configure(api_key=GOOGLE_API_KEY)
generative_model = genai.GenerativeModel(model_name='gemini-1.5-pro-latest')

# URL of the Wikipedia article (use any topic you prefer)
url = "https://en.wikipedia.org/wiki/Natural_language_processing"

# Fetch the page content
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Extract paragraphs from the article
text_data = []
for p in soup.find_all('p'):
    text = p.get_text(strip=True)
    text_data.append(text)

# Join all paragraphs into a single text
article_text = "\n\n".join(text_data)
print("Article content extracted.")


#chunking the data

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Split article text into chunks
chunks = text_splitter.split_text(article_text)

# Display chunks
for i, chunk in enumerate(chunks, start=1):
    print(f"Chunk {i}:\n{chunk}\n")

    #extracting key phrases

# Load spaCy's English model (make sure it's installed)
# You may need to run this once: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

# Define a function to extract keywords using noun chunks from spaCy
def extract_keywords_spacy(text):
    doc = nlp(text)
    return [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]  # Only keep phrases longer than 1 word

# Initialize a dictionary to store phrases for each chunk
chunk_phrases = {}

# Extract phrases for each chunk
for chunk_number, chunk in enumerate(chunks, start=1):
    phrases = extract_keywords_spacy(chunk)
    chunk_phrases[chunk_number] = phrases

# Display extracted phrases
for chunk_number, phrases in chunk_phrases.items():
    print(f"Key phrases from chunk {chunk_number}:\n{phrases}\n")

#generate embeddings for key phrases

def get_embedding(phrase):
    response = genai.embed_content(
        model='models/embedding-001',
        content=phrase,
        task_type="retrieval_query",
    )
    return response['embedding']

# Generate embeddings for each phrase
phrase_embeddings = {}
for chunk_number, phrases in chunk_phrases.items():
    embeddings = [get_embedding(phrase) for phrase in phrases]
    phrase_embeddings[chunk_number] = list(zip(phrases, embeddings))

# Prepare data for Excel output
excel_data = []
for chunk_number, phrases in phrase_embeddings.items():
    for phrase, embedding in phrases:
        excel_data.append({"Chunk": chunk_number, "Phrase": phrase, "Embedding": embedding})

# Save embeddings to Excel
df = pd.DataFrame(excel_data)
df.to_excel("phrases_embeddings_article.xlsx", index=False)
print("Embeddings saved to phrases_embeddings_article.xlsx")

#query processing and similarity calculation

def cosine_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)

query = "Explain the applications of NLP in healthcare."
query_phrases = [get_embedding(query)]
chunk_similarities = {}

# Calculate similarity for each chunk
for chunk_number, phrases in phrase_embeddings.items():
    similarities = []
    for phrase, embedding in phrases:
        phrase_similarities = [cosine_similarity(embedding, query_embedding) for query_embedding in query_phrases]
        similarities.append(max(phrase_similarities))
    chunk_similarities[chunk_number] = np.mean(similarities)

# Retrieve top 5 most relevant chunks
top_chunks = sorted(chunk_similarities.items(), key=lambda x: x[1], reverse=True)[:5]
selected_chunks = [chunks[chunk_number-1] for chunk_number, _ in top_chunks]
print("Top 5 relevant chunks:", selected_chunks)


#generate and retrieve answers with openai
context = "\n\n".join(selected_chunks)
prompt = f"Answer the following question based on the article:\n\n{context}\n\nQuestion: {query}\nAnswer:"

response = generative_model.generate_content(prompt)

answer = response.text.strip()
print(f"Answer:\n{answer}")
