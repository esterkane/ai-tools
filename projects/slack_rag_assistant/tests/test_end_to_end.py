import os
import json
import pytest
from app.slack_listener import app as slack_app
from app.embedding_service import getEmbeddings
from app.vector_store import VectorStore
from app.retriever import retrieveContext
from app.llm_interface import generateResponse

@pytest.fixture(scope='module')
def setup():
    # Load mock data
    with open(os.path.join('data', 'kb_articles.json')) as f:
        kb_articles = json.load(f)

    # Initialize vector store and FAISS index
    vector_store = VectorStore()
    vector_store.index_embeddings(kb_articles)

    yield vector_store

def test_end_to_end(setup):
    query = "What is the purpose of the RAG assistant?"
    expected_response = "The RAG assistant is designed to help users retrieve information from a knowledge base."

    # Generate embeddings for the query
    embeddings = getEmbeddings(query)

    # Retrieve context from the vector store
    context = retrieveContext(embeddings)

    # Generate response from the LLM
    response = generateResponse(query)

    assert expected_response in response

def test_slack_listener(setup):
    # Simulate a Slack event and assert the response
    event = {
        "type": "message",
        "text": "Tell me about the RAG assistant",
        "user": "U123456",
        "channel": "C123456"
    }

    response = slack_app.handle_event(event)

    assert response is not None
    assert "RAG assistant" in response['text']