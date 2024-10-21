
import bs4
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma   #This is latest way to import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings,HarmBlockThreshold, HarmCategory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import pyttsx3
import streamlit as st
import re
import os
from dotenv import load_dotenv
import speech_recognition as sr
import requests
from streamlit_lottie import st_lottie
import pickle
import logging

st.set_page_config(page_title="MaitriAI Chatbot", page_icon="image.png", layout="wide")
# Set up logger configuration
logging.basicConfig(
    level=logging.INFO,
    filename="log_file.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger=logging.getLogger()

# if logger.hasHandlers():
#         logger.handlers.clear()

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if API_KEY:
    logger.info("API key successfully loaded from environment.")
else:
    logger.error("Failed to load API key. Please check your .env file.")


# Initialize the model
try:
    

    llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    stream=True,
    safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
    
    )
    logger.info("Gemini model initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Gemini model: {e}")


# Load and split PDF document
try:
    pdf_loader = PyPDFLoader("MaitriAI_data_final1 (1).pdf")
    docs = pdf_loader.load()
    #st.write(docs)
    logger.info("PDF loaded and split successfully.")
except Exception as e:
    logger.error(f"Error loading/splitting PDF: {e}")


# Load or split document chunks
CHUNKS_FILE = "./document_chunks.pkl"

if os.path.exists(CHUNKS_FILE):
    with open(CHUNKS_FILE, "rb") as f:
        splits = pickle.load(f)
    logger.info("Loaded existing document chunks.")
else:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(splits, f)
    logger.info("Created and saved new document chunks.")


# Initialize embeddings and vector store
try:
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(documents=splits, embedding=gemini_embeddings)
    retriever = vectorstore.as_retriever()
    logger.info("Retriever initialized successfully.")
except ValueError as e:
    logger.error(f"Error initializing retriever: {e}")


# Set up contextualization and question-answer chains
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
) 
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

logger.info("History aware retriever created successfully. ")

qa_system_prompt = """
You are Alex, a MaitriAI assistant.
Start your conversation with asking user their name,email address and mobile number.
If user ask about dealing ,connecting or consulting related stuff.please assume they wanna do particular with respect to maitriAI.
MaitriAI is a company specialized in developing AI-driven applications that leverage the power of artificial intelligence to transform businesses. Our AI-driven application development services enable organizations to harness the potential of cutting-edge technologies and unlock new levels of efficiency, automation, and innovation.
Our main products are Credisence,AI Avatar,Customer Assistant Chatbot,LMS,AI Interviewer,OCR,Object detection based products amd much more.
Using advanced machine learning algorithms, natural language processing, computer vision, and data analytics, we create intelligent applications that can understand, learn, and adapt to complex business environments.
Provide concise answers using the following context. Limit responses to 2-3 sentences unless more detail is requested.
If the question isn't related to MaitriAI or its, respond with: "Sorry for the inconvenience, but I'm designed to answer your queries related to MaitriAI. What else can I assist you with?" also donot output any piece of text in bold font style.
{context} last but not the least if customer tends to be very aggressive or uses abusive or vulgur language donot indulge and politly ask to head to MaitriAI's official website. 


Finally you are smart assistant if something occurs that you are not capable to handle. you simply tell user to query at Contact section of company's website.
"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
try:
    logger.info("Retrieval Chain combined with question-answer chain")
except:

    logger.error(f"Retrieval chain Not Created.Either run entire code in new terminal or eliminate {e}.")




# Chat history management
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}
    if session_id not in st.session_state.chat_history:
        st.session_state.chat_history[session_id] = ChatMessageHistory()
    return st.session_state.chat_history[session_id]
logger.info("Previous conversations saved")
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Information extraction functions
def extract_info(text):
    name = extract_name(text)
    email = extract_email(text)
    mobile = extract_mobile(text)
    return name, email, mobile


def extract_name(text):
    name_pattern = r'\b[A-Z][a-zA-Z]+\b'
    names = re.findall(name_pattern, text)
    return " ".join(names[:2]) if names else None

def extract_mobile(text):
    mobile_pattern =  r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    mobile = re.search(mobile_pattern, text)
    return mobile.group() if mobile else None

def extract_email(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    email = re.search(email_pattern, text)
    return email.group() if email else None





# Helper functions
def clean_text(text):
    logger.info("Texts for pyttsx3 module is generating")
    cleaned_text = re.sub(r"[*\t]+", " ", text)
    logger.info("Texts for pyttsx3 module Generated and returned.")
    return cleaned_text


def text_2_speech_converter(text):
    logger.info("Text to speech function loaded.")
    engine = pyttsx3.init()
    statement2print = text.split(".")
    for statement in statement2print:
        new_text = clean_text(statement)
        engine.say(new_text)
        engine.runAndWait()


def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
        try:
            st.write("Recognizing...")
            text = recognizer.recognize_google(audio)
            logger.info(f"Speech recognized: {text}")
            response = conversational_rag_chain.invoke(
                {"input": text},
                config={"configurable": {"session_id": "MaitriAI_Test-II"}}
            )["answer"]
            st.write(response)
            text_2_speech_converter(response)
             # Extract information from user input
            name, email, mobile = extract_info(text)
            if name and email and mobile:
                print(f"Extracted Information: Name - {name}, Email - {email}, Mobile - {mobile}")
            return 1
        except sr.UnknownValueError:
            st.write("Sorry, I did not understand that.")
            logger.warning("Speech recognition failed: UnknownValueError")
            return 0
        except sr.RequestError:
            st.write("Could not request results; check your internet connection.")
            logger.error("Speech recognition failed: RequestError")
            return 0


def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        logger.warning(f"Failed to load Lottie animation from URL: {url}")
        return None
    return r.json()


# Main app function
def main():
    st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { color: #ffffff; background-color: #4CAF50; border-radius: 5px; }
    .stTextInput>div>div>input { border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar content
    with st.sidebar:
        st.image("Logo.png", width=200)
        st.title("MaitriAI")
        st.write("MaitriAI is a leading software company specializing in web & AI application development.")
    
    # Chat section
    col1, col2 = st.columns([1, 1])
    with col1:
        st.title("Chat with Alex 🤖")
        st.write("Ask me anything about MaitriAI's services!")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display all previous messages in the chat history
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["content"])
            elif message["role"] == "assistant":
                with st.chat_message("assistant"):
                    st.markdown(message["content"])

        # Input for current prompt
        prompt = st.chat_input("What would you like to know?")

        if prompt:
            # Save the user's question
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get the assistant's response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                response = conversational_rag_chain.invoke(
                    {"input": prompt},
                    config={"configurable": {"session_id": "MaitriAI_Test-II"}}
                )["answer"]
                message_placeholder.markdown(response)
            name, email, mobile = extract_info(prompt)
            if name and email and mobile:
                print(f"Extracted Information: Name - {name}, Email - {email}, Mobile - {mobile}")

                
            # Save the assistant's response
            st.session_state.messages.append({"role": "assistant", "content": response})


    with col2:
        lottie_url = "https://lottie.host/10a8e08e-4ec0-40db-b724-fb6a1ac4a411/eo4YYcDEIJ.json"
        lottie_bot = load_lottie_url(lottie_url)
        
        if lottie_bot:
            st_lottie(lottie_bot)
            logger.info("GIF from st.lottie loaded from URL")

        if st.button("Start Voice Input"):
            while speech_to_text():
                pass

main()
