import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Get the Google API key from environment variables
# The genai.configure call below uses os.getenv("GOOGLE_API_KEY") directly.
os.getenv("GOOGLE_API_KEY")
# Configure the Google Generative AI with the API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def extract_text_from_pdf(file_path):
    """
    Extracts text from a PDF file.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        str: The extracted text from the PDF.
    """
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() or "" # Handle potential None if page is empty
    except Exception as e:
        st.error(f"Error extracting text from PDF {file_path}: {e}")
        return ""
    return text

def extract_text_from_docx(file_path):
    """
    Extracts text from a DOCX file.

    Args:
        file_path (str): The path to the DOCX file.

    Returns:
        str: The extracted text from the DOCX.
    """
    text = ""
    try:
        doc = Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        st.error(f"Error extracting text from DOCX {file_path}: {e}")
        return ""
    return text

def extract_text_from_txt(file_path):
    """
    Extracts text from a TXT file.

    Args:
        file_path (str): The path to the TXT file.

    Returns:
        str: The extracted text from the TXT file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        st.error(f"Error extracting text from TXT {file_path}: {e}")
        return ""

def process_documents(file_paths):
    """
    Processes multiple document files (PDF, DOCX, TXT) and concatenates their text.

    Args:
        file_paths (list): A list of file paths to the documents.

    Returns:
        str: The combined text from all processed documents.

    Raises:
        ValueError: If an unsupported file type is encountered.
    """
    all_extracted_text = []

    for file_path in file_paths:
        file_extension = Path(file_path).suffix.lower()
        if file_extension == ".pdf":
            all_extracted_text.append(extract_text_from_pdf(file_path))
        elif file_extension == ".docx":
            all_extracted_text.append(extract_text_from_docx(file_path))
        elif file_extension == ".txt":
            all_extracted_text.append(extract_text_from_txt(file_path))
        else:
            # Instead of raising ValueError, use st.error for Streamlit and skip the file
            st.error(f"Unsupported file type for {file_path}. Skipping this file.")
            continue # Skip to the next file

    return "\n".join(filter(None, all_extracted_text)) # Filter out empty strings from failed extractions

def get_text_chunks(text):
    """
    Splits a given text into smaller chunks for processing.

    Args:
        text (str): The input text to be chunked.

    Returns:
        list: A list of text chunks.
    """
    # Initialize RecursiveCharacterTextSplitter with chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """
    Creates a FAISS vector store from text chunks using Google Generative AI Embeddings
    and saves it locally.

    Args:
        text_chunks (list): A list of text chunks.
    """
    if not text_chunks:
        st.warning("No text chunks to create a vector store. Please ensure documents have content.")
        return

    try:
        # Initialize embeddings model
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # Create FAISS vector store from text chunks and embeddings
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        # Save the vector store locally
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error creating or saving vector store: {e}")

def get_conversational_chain():
    """
    Creates and returns a conversational chain using Google Generative AI.

    Returns:
        langchain.chains.question_answering.load_qa_chain: The conversational chain.
    """
    # Define the prompt template for the conversational AI. This example is for a technical advisor chatbot.
    prompt_template = """
    You are a helpful Technical Advisor Chatbot.
    Answer the question as detailed as possible from the provided context. When answering the question, ignore any product number starting with 9 and its context.
    If the answer is not in the provided context, just say, "The answer is not available. Please seek advice from your Account Manager" Do not provide wrong answers.

    Chat History:
    {chat_history}

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    # Initialize the ChatGoogleGenerativeAI model. Using gemini-2.5-flash
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    # Create a PromptTemplate with specified input variables
    prompt = PromptTemplate(template=prompt_template, input_variables=["chat_history", "context", "question"])
    # Load the QA chain with the model and prompt
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, chat_history):
    """
    Processes the user's question, performs a similarity search on the vector store,
    and gets a response from the conversational AI.

    Args:
        user_question (str): The question asked by the user.
        chat_history (list): A list of previous chat messages.

    Returns:
        str: The AI's response to the user's question.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Check if faiss_index exists before loading
    if not os.path.exists("faiss_index"):
        st.error("No documents processed yet. Please upload and process documents first.")
        return "Please upload and process documents before asking questions."

    try:
        # Load the FAISS vector store
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}. Please try re-processing documents.")
        return "An error occurred while loading documents. Please re-process them."

    # Convert chat_history list of dicts to a string format for the prompt
    history_string = ""
    # Only include previous user/assistant turns in history_string
    for msg in chat_history:

        if msg["role"] in ["user", "assistant"]:
            history_string += f"{msg['role'].capitalize()}: {msg['content']}\n"

    # Combine chat history for context with the user question
    # This ensures the similarity search considers the conversation flow
    search_query = history_string + "\n" + user_question

    # Perform similarity search to retrieve relevant documents
    # Limit the number of documents to retrieve for efficiency and relevance
    docs = new_db.similarity_search(search_query, k=5)

    # Get the conversational chain
    chain = get_conversational_chain()

    try:
        # Get the response from the chain
        response = chain(
            {"input_documents": docs, "question": user_question, "chat_history": history_string},
            return_only_outputs=True
        )
        return response["output_text"]
    except Exception as e:
        st.error(f"Error generating response from AI model: {e}")
        return "An error occurred while generating a response. Please try again."

def main():
    """
    Main function to run the Streamlit application.
    Sets up the UI, handles document uploads, processing, and chat interactions.
    """
    st.set_page_config(
        page_title="Technical Advisor",
        page_icon="📚",
        layout="wide"
    )
    st.header("Technical Advisor Chatbot 👨‍🔧")

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize a flag to indicate if a bot response is being generated
    if "awaiting_bot_response" not in st.session_state:
        st.session_state.awaiting_bot_response = False

    # Display chat history from session state
    # Ensure messages are displayed even if awaiting a bot response
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input at the bottom of the main page
    # The chat_input widget is disabled when awaiting a bot response
    if user_question := st.chat_input("Type your message...", disabled=st.session_state.awaiting_bot_response):
        # Prevent adding duplicate user messages if a rerun occurs
        if not st.session_state.messages or \
           not (st.session_state.messages[-1]["role"] == "user" and \
                st.session_state.messages[-1]["content"] == user_question):
            st.session_state.messages.append({"role": "user", "content": user_question})
            st.session_state.awaiting_bot_response = True
            st.rerun() # Rerun immediately to display user message and then process bot response

    # --- Bot Response Generation Logic ---
    # This block executes if awaiting_bot_response is True, meaning a user question was just submitted
    if st.session_state.awaiting_bot_response:
        # Ensure the last message is from the user and needs a response
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    current_user_q = st.session_state.messages[-1]["content"]
                    # Pass the current chat history (including the just-added user question)
                    bot_response = user_input(current_user_q, list(st.session_state.messages))

            # Append bot response to history only once, outside the spinner context
            st.session_state.messages.append({"role": "assistant", "content": bot_response})

            # Reset the flag and rerun to update the UI (clear spinner, re-enable input)
            st.session_state.awaiting_bot_response = False
            st.rerun() # Rerun to display the bot's final message and re-enable chat input

    # Sidebar for document upload and processing
    with st.sidebar:
        st.title("Menu:")
        # File uploader allows multiple files of specified types
        documents = st.file_uploader(
            "Upload your documents (.pdf, .docx, .txt) and Click on the Submit & Process Button",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt"] # Explicitly define accepted types
        )
        file_paths = []
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True) # Create data directory if it doesn't exist

        # If documents are uploaded, save them to the 'data' directory
        if documents:
            for doc in documents:
                # Create a safe file path using Path.joinpath
                file_path = data_dir.joinpath(doc.name)
                try:
                    with open(file_path, "wb") as f:
                        f.write(doc.getbuffer())
                    file_paths.append(str(file_path))
                except Exception as e:
                    st.error(f"Error saving {doc.name}: {e}")

        # Button to trigger document processing
        if st.button("Submit & Process Documents"):
            if file_paths:
                with st.spinner("Processing documents..."):
                    raw_text = process_documents(file_paths)
                    if raw_text: # Only proceed if text was successfully extracted
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Documents Processed Successfully")
                    else:
                        st.warning("No text extracted from the uploaded documents. Please check file contents.")
            else:
                st.warning("Please upload documents before processing.")

            # Clean up uploaded files after processing to save space and prevent re-processing on rerun
            # You might want to keep them if you intend to re-process without re-uploading,
            # but for a typical RAG app, deleting them after processing to FAISS is common.
            # for file_path in file_paths:
            #     try:
            #         os.remove(file_path)
            #     except OSError as e:
            #         print(f"Error deleting file {file_path}: {e}")

if __name__ == "__main__":
    # Command to run the Streamlit app: streamlit run src/app.py
    # Make sure you have a .env file with GOOGLE_API_KEY="YOUR_API_KEY" in the same directory as the script.
    main()