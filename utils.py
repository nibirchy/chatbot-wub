import os
import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize GPT-Neo with controlled parameters
generator = pipeline(
    "text-generation",
    model="distilgpt2",
    max_length=100,
    pad_token_id=50256
)

# Initialize Pinecone client
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_client = Pinecone(api_key=pinecone_api_key, environment='us-east-1-aws')

def truncate_text(text, max_length=512):
    """Truncate text to prevent exceeding model's context window"""
    words = text.split()
    if len(words) > max_length:
        return ' '.join(words[:max_length])
    return text

def query_refiner(conversation, query):
    """Refine the query with length constraints"""
    try:
        truncated_input = truncate_text(
            f"Given the following conversation and query, refine the question:\n\n"
            f"CONVERSATION LOG: \n{conversation[-200:]}\n\n"
            f"Query: {query}\n\nRefined Query:"
        )
        refined_query = generator(
            truncated_input,
            max_new_tokens=50,
            do_sample=True
        )[0]['generated_text']
        return refined_query.strip()
    except Exception as e:
        print(f"Error in query_refiner: {str(e)}")
        return query  # Return original query if refinement fails

def find_match(input):
    """Find relevant matches in Pinecone with error handling"""
    try:
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        input_em = model.encode(truncate_text(input, 128)).tolist()

        # Query Pinecone with error handling
        result = pinecone_client.Index("langchain-chatbot").query(
            vector=input_em,
            top_k=2,
            include_metadata=True
        )

        if not result['matches']:
            return "No relevant information found."

        # Combine results with length limitation
        texts = []
        total_length = 0
        max_length = 500  # Maximum combined length

        for match in result['matches']:
            text = match['metadata'].get('text', '')
            if total_length + len(text) <= max_length:
                texts.append(text)
                total_length += len(text)
            else:
                break

        return "\n".join(texts) if texts else "No relevant information found."

    except Exception as e:
        print(f"Error in find_match: {str(e)}")
        return "Error retrieving information."

def get_conversation_string():
    """Get conversation history with length limitation"""
    try:
        conversation_string = ""
        max_entries = 5  # Limit the number of conversation pairs to include
        
        responses = st.session_state.get('responses', [])
        requests = st.session_state.get('requests', [])
        
        # Take only the last few conversations
        start_idx = max(0, len(responses) - max_entries - 1)
        
        for i in range(start_idx, len(responses) - 1):
            if i < len(requests):
                conversation_string += f"Human: {requests[i][:100]}\n"  # Limit length of each entry
                conversation_string += f"Bot: {responses[i + 1][:100]}\n"
        
        return conversation_string
    except Exception as e:
        print(f"Error in get_conversation_string: {str(e)}")
        return ""