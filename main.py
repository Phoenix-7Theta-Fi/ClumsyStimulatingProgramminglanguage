import streamlit as st
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage

# Streamlit app setup
st.set_page_config(page_title="Ayurfix - Ayurvedic Consultation App")
st.title("Ayurfix - Your AI Ayurvedic Consultant")

# Load secrets
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
NEO4J_URI = st.secrets["NEO4J_URI"]
NEO4J_USERNAME = st.secrets["NEO4J_USERNAME"]
NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]

# Check if all required credentials are available
if not all([GOOGLE_API_KEY, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
    st.error("Missing required credentials. Please check your secrets.toml file.")
    st.stop()

# Connect to Neo4j
try:
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD
    )
except Exception as e:
    st.error(f"Failed to connect to Neo4j: {str(e)}")
    st.stop()

# Initialize Google Gemini Pro model
try:
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"Failed to initialize Google Gemini Pro model: {str(e)}")
    st.stop()

# Set up conversation memory
conversation_memory = ConversationBufferMemory(input_key='human_input', memory_key='chat_history')

# Create GraphCypherQAChain
try:
    graph_qa_chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True
    )
except Exception as e:
    st.error(f"Failed to create GraphCypherQAChain: {str(e)}")
    st.stop()

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
        try:
            graph_response = graph_qa_chain.run(user_input)

            # Use the LLM to generate a more comprehensive response
            llm_prompt = prompt.format(
                human_input=user_input,
                graph_info=graph_response,
                chat_history="\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
            )
            llm_response = llm([HumanMessage(content=llm_prompt)])

            # Display AI's response
            with st.chat_message("assistant"):
                st.markdown(llm_response.content)
            st.session_state.messages.append({"role": "assistant", "content": llm_response.content})
        except Exception as e:
            error_message = str(e)
            if "API_KEY_INVALID" in error_message:
                st.error("The Google API key is invalid or has expired. Please contact the administrator to update the API key.")
            else:
                st.error(f"An error occurred while generating the response: {error_message}")

# Disclaimer
st.markdown("---")
st.write("Disclaimer: This app provides general Ayurvedic information and is not a substitute for professional medical advice. Always consult with a qualified healthcare provider for personalized medical guidance.")

# Run the Streamlit app
if __name__ == "__main__":
    st.write("Ayurfix is ready to assist you with Ayurvedic consultations!")