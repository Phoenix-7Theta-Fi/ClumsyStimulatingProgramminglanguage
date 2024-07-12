import streamlit as st
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Streamlit app setup
st.set_page_config(page_title="Ayurfix - Ayurvedic Consultation App")
st.title("Ayurfix - Your AI Ayurvedic Consultant")

# Load secrets
NEO4J_URI = st.secrets["neo4j"]["uri"]
NEO4J_USERNAME = st.secrets["neo4j"]["username"]
NEO4J_PASSWORD = st.secrets["neo4j"]["password"]
GOOGLE_API_KEY = st.secrets["google"]["api_key"]

# Connect to Neo4j
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

# Initialize Google Gemini Pro model
llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

# Set up conversation memory
conversation_memory = ConversationBufferMemory(input_key='human_input', memory_key='chat_history')

# Create GraphCypherQAChain
graph_qa_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True
)

# Ayurvedic consultation prompt template
ayurvedic_template = """
You are an AI Ayurvedic practitioner named Ayurfix. Use the following steps to provide an Ayurvedic consultation:

1. Analyze the user's input: {human_input}
2. Use the Neo4j graph database to find relevant Ayurvedic concepts, treatments, and recommendations.
3. Provide a consultation based on Ayurvedic principles, including:
   - Potential Dosha imbalances
   - Recommended treatments or therapies
   - Dietary and lifestyle advice
   - Any relevant ingredients or herbs
4. If you don't have enough information, ask follow-up questions to gather more details about the user's condition or concerns.

Use this information from the knowledge base to support your recommendations:
{graph_info}

Current conversation:
{chat_history}

Provide a comprehensive and compassionate response.
"""

prompt = PromptTemplate(
    input_variables=['human_input', 'graph_info', 'chat_history'],
    template=ayurvedic_template
)

# Streamlit app layout
st.write("Welcome to Ayurfix! I'm your AI Ayurvedic consultant. Please describe your symptoms, concerns, or ask any questions about Ayurveda.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if user_input := st.chat_input("What would you like to know about Ayurveda?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response using the graph QA chain
    with st.spinner("Thinking..."):
        graph_response = graph_qa_chain.run(user_input)

        # Use the LLM to generate a more comprehensive response
        llm_response = llm.generate_content(
            prompt.format(
                human_input=user_input,
                graph_info=graph_response,
                chat_history="\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
            )
        )

    # Display AI's response
    with st.chat_message("assistant"):
        st.markdown(llm_response.text)
    st.session_state.messages.append({"role": "assistant", "content": llm_response.text})

# Disclaimer
st.markdown("---")
st.write("Disclaimer: This app provides general Ayurvedic information and is not a substitute for professional medical advice. Always consult with a qualified healthcare provider for personalized medical guidance.")

# Run the Streamlit app
if __name__ == "__main__":
    st.write("Ayurfix is ready to assist you with Ayurvedic consultations!")