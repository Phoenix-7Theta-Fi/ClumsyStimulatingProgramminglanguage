import streamlit as st
import os
from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI

# Set page config
st.set_page_config(page_title="Ayurvedic Assistant", page_icon="🌿", layout="wide")

# Set up Google API key
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Connect to Neo4j
@st.cache_resource
def init_graph():
    return Neo4jGraph(
        url=st.secrets["NEO4J_URI"],
        username=st.secrets["NEO4J_USERNAME"],
        password=st.secrets["NEO4J_PASSWORD"]
    )

graph = init_graph()

# Initialize Google Gemini Pro
@st.cache_resource
def init_llm():
    return GoogleGenerativeAI(model="gemini-pro", temperature=0.1)

llm = init_llm()

# Create a memory object to store conversation history
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define a custom prompt template for Ayurvedic diagnosis
ayurvedic_prompt = PromptTemplate(
    template="""You are an Ayurvedic expert assistant. Use the following pieces of context to answer the human's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}