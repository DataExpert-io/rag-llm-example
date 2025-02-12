import os
from flask import Flask, request, jsonify
from openai import OpenAI
from pinecone import Pinecone
app = Flask(__name__)

# --- Configure your API keys ---
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# --- Initialize Pinecone ---
pinecone_client = Pinecone(
    api_key=PINECONE_API_KEY
)
INDEX_NAME = 'rag-example-index'  # Replace with your Pinecone index name
# Assuming the index has already been created in Pinecone
index = pinecone_client.Index(INDEX_NAME)
CHATGPT_MODEL = "gpt-4"


def query_pinecone(query_embedding, top_k=20):
    """Queries Pinecone to retrieve top_k most similar contexts to the query."""
    query_response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_values=True,  # Only need metadata, unless you want the vectors
        include_metadata=True  # So we can retrieve text or other data
    )
    return query_response


# --- Helper function to get embeddings from OpenAI ---
def get_openai_embedding(text, model="text-embedding-ada-002"):
    """Generate embedding for the text using OpenAI."""
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding


# --- Helper function to call ChatGPT with context ---
def generate_answer_with_context(question, context):
    """Calls ChatGPT with the retrieved context and returns an answer."""
    # Prepare a prompt with the context
    system_content = (
        "You are a helpful assistant. Use the following context to answer the question, make sure to include the file path and repo in your response:\n\n"
        f"{context}\n\n"
        "If the answer cannot be found in the context, provide your best possible answer."
    )

    print(system_content)
    print(question)

    response = client.chat.completions.create(
        model=CHATGPT_MODEL,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": question}
        ],
        temperature=0.1,
        max_tokens=500
    )
    print(response)

    answer = response.choices[0].message.content
    return answer


@app.route("/ask", methods=["GET"])
def ask_question():
    question = request.args.get('question')
    """
    Expects a JSON body: {"question": "Your question"}
    Returns a JSON object with the answer.
    """
    if not question:
        return jsonify({"error": "No question provided"}), 400

    # 1. Embed the question
    question_embedding = get_openai_embedding(question)
    print('embedding is')
    print(question_embedding)
    # 2. Query Pinecone for relevant documents
    pinecone_results = query_pinecone(question_embedding, top_k=20)
    print(pinecone_results)
    # 3. Extract relevant context from the metadata (assuming we stored text in metadata["text"])
    retrieved_contexts = []
    for match in pinecone_results.matches:
        metadata = match.metadata
        print(metadata)
        text = metadata.get("content", "")
        repo = metadata.get('repo', '')
        file = metadata.get('file', '')
        combined_text = text
        if repo or file:
            combined_text = f"""Found in file: {file}, repo: {repo} content: {text}"""
        retrieved_contexts.append(combined_text)

    print(retrieved_contexts)
    # Combine the retrieved contexts into one string
    combined_context = "\n\n".join(retrieved_contexts)

    # 4. Send the question + context to ChatGPT
    answer = generate_answer_with_context(question, combined_context)

    # 5. Return the answer
    return jsonify({"question": question, "answer": answer})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3001)