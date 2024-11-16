from flask import Flask, request, jsonify
import requests
import numpy as np
import faiss
import json
import google.generativeai as genai

# Initialize Flask app
app = Flask(__name__)

# Load your Gemini API key
GEMINI_API_KEY = 'AIzaSyCsrbKwINRc9_NZtT0kuCKdo5uggzVQvIE'
genai.configure(api_key=GEMINI_API_KEY)

HEADERS = {
    "Content-Type": "application/json"
}

GEMINI_COMPLETION_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

# Load the dataset and chunk it
data = None
chunks = None
embedding_matrix = None
index = None

# Function to load large data (for example, from a file)
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to chunk text
def chunk_data(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# Function to get embeddings using Gemini API
def get_embeddings_from_gemini(texts):
    embeddings = []
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=texts,
            task_type="retrieval_document"
        )
        for embedding in result['embedding']:
            embeddings.append(embedding)
    except Exception as e:
        print(f"Error fetching embeddings: {e}")
    return embeddings

# Function to generate a response from the model
def generate_response(prompt):
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    try:
        response = requests.post(
            GEMINI_COMPLETION_URL,
            headers=HEADERS,
            data=json.dumps(payload)
        )
        response.raise_for_status()
        data = response.json()
        if 'candidates' in data and len(data['candidates']) > 0:
            candidate = data['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content']:
                parts = candidate['content']['parts']
                if len(parts) > 0 and 'text' in parts[0]:
                    return parts[0]['text']
        return "No valid response found."
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

# Function to retrieve relevant chunks based on a query
def retrieve_relevant_chunks(query):
    query_embedding = get_embeddings_from_gemini([query])[0]
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, k=1)
    closest_chunk_index = indices[0][0]
    return chunks[closest_chunk_index]

# Initialize dataset, chunks, embeddings, and FAISS index
def initialize_data():
    global data, chunks, embedding_matrix, index
    data = load_data('../data.txt')
    chunks = chunk_data(data)
    embeddings = get_embeddings_from_gemini(chunks)
    embedding_matrix = np.array(embeddings).astype('float32')
    dimension = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_matrix)
    print("Data initialized.")

# Root endpoint
@app.route('/')
def home():
    return "Welcome to the Flask API! Use the '/query' endpoint to interact with the model."

# Endpoint to handle queries
@app.route('/query', methods=['POST'])
def handle_query():
    user_query = request.json.get('query')
    if not user_query:
        return jsonify({'error': 'No query provided'}), 400

    relevant_chunk = retrieve_relevant_chunks(user_query)
    prompt = f"Context: {relevant_chunk}\n\nUser Query: {user_query}\n\nProvide a detailed explanation based on the context."
    response = generate_response(prompt)

    if response:
        return jsonify({'response': response})
    else:
        return jsonify({'error': 'Failed to generate a response'}), 500

# Initialize data before the server starts
initialize_data()

# Run the Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
