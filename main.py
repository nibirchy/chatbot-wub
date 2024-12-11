import os
import streamlit as st
from transformers import pipeline
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from streamlit_chat import message
from utils import query_refiner, find_match, get_conversation_string
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as LangChainPinecone
from langchain.llms import HuggingFacePipeline
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file
load_dotenv()

print("PINECONE_API_KEY:", os.getenv("PINECONE_API_KEY"))

# Streamlit app title
st.title("Bangladeshi Women's Legal Help Chatbot")

# Initialize session state variables if they don't exist
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

# Initialize Hugging Face pipeline with max length constraint
generator = pipeline(
    "text-generation",
    model="distilgpt2",
    max_length=100,  # Limit the total length
    pad_token_id=50256  # GPT-2's pad token ID
)

# Wrap the pipeline into a LangChain-compatible LLM
llm = HuggingFacePipeline(pipeline=generator)

# Initialize Pinecone client
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment='us-east-1-aws')

# Create Pinecone index if it doesn't exist
index_name = 'langchain-chatbot'
if index_name not in pinecone_client.list_indexes().names():
    pinecone_client.create_index(
        name=index_name,
        dimension=384,  # Update dimension for BanglaBERT embeddings
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Initialize Pinecone index and LangChain wrapper
index = pinecone_client.Index(index_name)
embeddings = HuggingFaceEmbeddings(model_name='sagorsarker/bangla-bert-base')
vector_store = LangChainPinecone.from_documents([], embeddings, index_name=index_name)

# Set up conversation memory with LangChain
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

# Set up conversation chain
system_msg_template = SystemMessagePromptTemplate.from_template(
    template="""Answer the question as truthfully as possible using the provided context, 
    and if the answer is not contained within the text below, say 'I don't know'"""
)
human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
prompt_template = ChatPromptTemplate.from_messages([
    system_msg_template,
    MessagesPlaceholder(variable_name="history"),
    human_msg_template
])

conversation = ConversationChain(
    memory=st.session_state.buffer_memory,
    prompt=prompt_template,
    llm=llm,
    verbose=True
)

# Containers for chat interface
response_container = st.container()
textcontainer = st.container()

# Function to truncate input text
def truncate_text(text, max_length=512):
    words = text.split()
    if len(words) > max_length:
        return ' '.join(words[:max_length])
    return text

# Handling user input and generating responses
with textcontainer:
    query = st.text_input("Ask Question: ", key="input")
    if query:
        try:
            with st.spinner("Thinking..."):
                # Truncate and process the query
                truncated_query = truncate_text(query)
                conversation_string = get_conversation_string()

                # Skip query refinement for now
                refined_query = truncated_query

                # Retrieve context from Pinecone
                context = find_match(refined_query)

                # Generate response
                response = generator(context, max_new_tokens=100, do_sample=True)[0]["generated_text"]

                # Format the response for UI with aligned new lines
                formatted_response = response.replace("\n", "\n\n")

                # Append response to session state
                st.session_state.requests.append(truncated_query)
                st.session_state.responses.append(formatted_response)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.session_state.responses.append("I apologize, but I encountered an error. Please try rephrasing your question.")

# Displaying the conversation
with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')

import os
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone client
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_client = Pinecone(api_key=pinecone_api_key, environment='us-east-1-aws')

# Load BanglaBERT model and tokenizer
banglabert_tokenizer = AutoTokenizer.from_pretrained('sagorsarker/bangla-bert-base')
banglabert_model = AutoModel.from_pretrained('sagorsarker/bangla-bert-base')

def truncate_text(text, max_length=512):
    """Truncate text to prevent exceeding model's context window"""
    words = text.split()
    if len(words) > max_length:
        return ' '.join(words[:max_length])
    return text

def query_refiner(conversation, query):
    """Temporarily disabled query refinement"""
    return query.strip()

def encode_text(text):
    """Encode text using BanglaBERT"""
    inputs = banglabert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = banglabert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

def find_match(input):
    """Find relevant matches in Pinecone with BanglaBERT embeddings"""
    try:
        input_em = encode_text(truncate_text(input, 128))

        result = pinecone_client.Index("langchain-chatbot").query(
            vector=input_em,
            top_k=5,  # Retrieve more matches
            include_metadata=True
        )

        # Filter matches by a minimum similarity score
        threshold = 0.5
        matches = [match for match in result['matches'] if match['score'] >= threshold]

        if not matches:
            return "No relevant information found."

        # Combine texts with a length constraint
        texts = []
        total_length = 0
        max_length = 500

        for match in matches:
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
