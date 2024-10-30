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
import mysql.connector
from mysql.connector import Error
from typing import Optional
from langchain_core.pydantic_v1 import BaseModel, Field








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
API_KEY = os.environ["GOOGLE_API_KEY"]

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
    vectorstore = Chroma.from_documents(documents=splits, embedding=gemini_embeddings,persist_directory="./chroma_langchain_db")
    retriever = vectorstore.as_retriever( search_type="similarity",search_kwargs={'k': 6})
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



#


qa_system_prompt = """
You are Alex, a professional assistant for MaitriAI.
Start the conversation by politely asking for the user's name, email address, and mobile number if possible and they doesnt provide just ask for atleast email . Inform the user that if they wish to update any of these three details later, they must update all three for consistency.
If user

If the user inquires about connecting, consulting, or engaging in business with MaitriAI, assume their interest is specifically regarding MaitriAI’s offerings, and respond accordingly.

MaitriAI is a company specializing in AI-driven applications that enhance business efficiency and innovation through artificial intelligence. Our AI services include products like Credisence, AI Avatar, Customer Assistant Chatbot, LMS, AI Interviewer, OCR, Object Detection-based products, and more.

Using machine learning, natural language processing, computer vision, and data analytics, MaitriAI develops applications that can adapt to complex business environments. Provide concise answers with 2-3 sentences unless the user requests further detail.

**Guidelines for Responses:**
- **Clarifying Intent:** If a user’s question seems unclear or lacks detail, kindly ask them for more specifics. For instance: "Could you provide a bit more detail so I can assist you accurately?"
- **Professional Empathy:** If a user appears frustrated or expresses dissatisfaction, respond professionally with empathy, e.g., "I understand your concern. I'm here to help and ensure you get the assistance you need. Could you clarify how I can assist further?"
- **Ambiguity and Consistency:** If a question is too broad or general, provide a brief overview and offer to elaborate on specific areas if the user is interested.
- **Non-MaitriAI Queries:** If the question is unrelated to MaitriAI, respond with: "I'm here to assist with queries related to MaitriAI. Could I help with something specific about our services?"

Lastly, if the user uses inappropriate language or becomes aggressive, politely suggest they visit MaitriAI’s official website or mail at contact@maitriai.com. for further assistance.

If a query is outside your capabilities, suggest they visit the "Contact" section on MaitriAI’s website.
{context}
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


from typing import Optional,Dict

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field


def create_connection():
    try:
        logger.info('Entered inside create_connection()')
        connection = mysql.connector.connect(
            host="localhost",  # or your host
            user="root",       # your MySQL username
            password="",  # your MySQL password
            database=" "  # your database
        )
        if connection.is_connected():
            logger.info("Database Connection Created")
            return connection
    except Error as e:
        logger.error(f"Error connecting to MySQL: {e}")
        return None

def check_user_exists(email):
    # Right now im filterning wrt email
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            query = "SELECT COUNT(*) FROM users WHERE email = %s"
            cursor.execute(query, (email,))
            result = cursor.fetchone()
            return result[0] > 0  # Return True if the user exists
        except Error as e:
            logger.error(f"Error checking user existence: {e}")
            return False
        finally:
            cursor.close()
            connection.close()
    return False




def insert_user_info(name, email, mobile):
    # Check if the user already exists before inserting
   
    if not check_user_exists(email):  # Create this function
        connection = create_connection()
        if connection:
            try:
                cursor = connection.cursor()
                query = """INSERT INTO users (name, email, mobile) VALUES (%s, %s, %s)"""
                cursor.execute(query, (name, email, mobile))
                connection.commit()
                logger.info(f"User info inserted successfully: {name}, {email}, {mobile}")
            except Error as e:
                logger.error(f"Error inserting user info: {e}")
            finally:
                cursor.close()
                connection.close()
    else:
        logger.info("User already exists; skipping insert.")









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


def details_extraction(text):
    """Extract contact details from text."""
    logger.info("Entered to details_extraction function")
    class Person(BaseModel):
        """Extract contact details from text."""
        
        name: str = Field(
            default=None, 
            description="Extract full name of the person from text,IF EXISTS",
            
            
        )
        mobile: str = Field(
            default="", 
            description="Extract mobile/phone number from text if present,IF EXISTS",
            
            
        )
        email_id: str = Field(
            default="", 
            description="Extract email address from text if present,IF EXISTS",
           
        )
    logger.info("CROSSED THE SCHEMA FUNCTION")
    
    # Improved prompt for better extraction
    info_extract_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert information extraction algorithm.
            Your task is to extract name, mobile number, and email address of person if present from the input text.
            
            If any information is not present, return null for that field."""
        ),
        ("human", "{text}"),
    ])
    llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,

    
    )
    
    runnable = info_extract_prompt | llm.with_structured_output(schema=Person)
    
    
    result = runnable.invoke({"text": text})
    logger.info(f"Extraction result: {result}")
    print(result.name, result.mobile, result.email_id)
    return result.name, result.mobile, result.email_id



import os
import json
from datetime import datetime

def save_conversation_to_file(email: str, conversation: list) -> str:
    """
    Save conversation to a JSON file and return the file path.
    """
    # Create a conversations directory if it doesn't exist
    os.makedirs('conversations', exist_ok=True)
    
    # Create a timestamp-based filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"conversations/{email}_{timestamp}.json"
    
    # Format the conversation data
    conversation_data = {
        'email': email,
        'timestamp': timestamp,
        'messages': conversation
    }
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(conversation_data, f, indent=2)
    
    return filename
# def update_database_schema():
#     """
#     Update database schema to include conversation_file column in MySQL.
#     """
#     connection = create_connection()  # Get database connection
#     if connection:
#         try:
#             cursor = connection.cursor()
            
#             # Step 1: Check if column exists
#             check_column_query = """
#             SELECT COUNT(*)
#             FROM information_schema.columns 
#             WHERE table_schema = 'maitriai_db'
#             AND table_name = 'users' 
#             AND column_name = 'conversation_file';
#             """
#             cursor.execute(check_column_query)  # Execute the check
#             column_exists = cursor.fetchone()[0]  # Get the result
            
#             # Step 2: Only add column if it doesn't exist
#             if column_exists == 0:
#                 alter_query = """
#                 ALTER TABLE users 
#                 ADD COLUMN conversation_file VARCHAR(255)
#                 """
#                 cursor.execute(alter_query)  # Execute the alter
#                 connection.commit()  # Commit the changes
#                 logger.info("conversation_file column added successfully")
#             else:
#                 logger.info("conversation_file column already exists")
                
#         except Error as e:
#             logger.error(f"Error updating database schema: {e}")
#         finally:
#             cursor.close()
#             connection.close()



def update_user_conversation(email: str, conversation_file: str):
    """
    Update the conversation file path in the database for a user.
    """
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            query = """
            UPDATE users 
            SET conversation_file = %s 
            WHERE email = %s
            """
            cursor.execute(query, (conversation_file, email))
            connection.commit()
            logger.info(f"Updated conversation file path for user: {email}")
        except Error as e:
            logger.error(f"Error updating conversation file path: {e}")
        finally:
            cursor.close()
            connection.close()

    
# Main app function
def main():
    #update_database_schema()
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
        if "user_info" not in st.session_state:
            st.session_state.user_info = {
                "name": None,
                "mobile": None,
                "email": None
            }

        # Display all previous messages in the chat history
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["content"])
            elif message["role"] == "assistant":
                with st.chat_message("assistant"):
                    st.markdown(message["Response"])

        # Input for current prompt
        prompt = st.chat_input(placeholder="Ask Anything")

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
                    config={"configurable": {"session_id": "MaitriAI_Test-III"},
                           
                            
                            }
                )["answer"]
            

                message_placeholder.markdown(response)

                st.session_state.messages.append({"role": "assistant", "Response": response})
                

            try:
                name, mobile, email_id = details_extraction(prompt)
                conversation_file = save_conversation_to_file(email_id, st.session_state.messages)
                # Update database with file path
                
                update_user_conversation(email_id, conversation_file)
                if email_id:  # Only save if we have an email
                    conversation_file = save_conversation_to_file(email_id, st.session_state.messages)
                    update_user_conversation(email_id, conversation_file)
                    logger.info(f'Conversation saved for user with email: {email_id}')
                
                if name!=None:
                    #st.write(f"Name: {name}")
                    logger.info('Name Extracted Successfully')
                if mobile:
                    #st.write(f"Mobile:{mobile}")
                    logger.info('Mobile Number Extracted Successfully')
                if email_id:
                    #st.write(f"Email: {email_id}")
                    logger.info('Name Extracted Successfully')

                
                insert_user_info(name,email_id,mobile)




                # for ele in st.session_state.messages:
                #     st.write(ele)
                
            except Exception as e:
                logger.error(f"Error extracting details: {str(e)}")
                st.error("Could not extract contact information")




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

