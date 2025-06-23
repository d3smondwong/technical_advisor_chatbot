# RAG Chatbot
Streamlit application that processes documents (PDF, DOCX, TXT), creates a FAISS vector store for retrieval-augmented generation (RAG), and interacts with a LLM (`Gemini 2.5 Flash` in this example) to answer user questions based on the uploaded documents.

### Use Case (Technical Advisor Chatbot)
When I first joined Wilhelmsen Ships Service as an Account Manager, I was overwhelmed by the sheer number of products we offered. Each product has its own unique use case, application method, control guidelines, and dosages requirements in the case of marine chemicals. It can be quite daunting for new hires to navigate all of that.

Thankfully, I received constant support from our Technical Sales Managers, who were always available to help with technical questions and learn from them over the years. But like many specialists in a fast-paced industry, their calendars were packed—customer meetings, site visits, urgent calls. This sometimes meant delays in getting answers to customer queries, especially from Technical Superintendents or Chief Engineers. And in the fast-paced world of shipping and logistics, timeliness truly matters.

With the benefit of hindsight, I started asking myself: How could we use AI to support roles like mine and improve the customer experience?

That question led me to build a chatbot that uses AI to search through Wilhelmsen’s product catalogue and technical manuals (available publicly). It's designed to help quickly surface technical answers—everything from usage instructions to troubleshooting guidance.

Could this be a helpful co-pilot for Account Managers and Customer Service teams?

Or perhaps, an always-available Technical Advisor for Wilhelmsen customers, 24/7?

**Use Case example**

![Technical Advisor Chatbot](demo_assets\WSS Technical Advisor.mp4)

## Contents
- Folder Structure
- Core Functionality
- How to run this application?
- How to use the application?

### Folder Structure
```
.
├── .env
├── .gitignore
├── README.md
├── requirements.txt
├── src
|   └── app.py
├── data
|   └── (Your uploaded documents will be saved here, e.g., document1.pdf, report.docx)
└── faiss_index
    ├── index.faiss
    └── index.pkl
```

### Core Functionality

1. Document Extraction:

    `extract_text_from_pdf`: Uses PyPDF2 to read text from PDF files.

    `extract_text_from_docx`: Uses python-docx to read text from DOCX files.

    `extract_text_from_txt`: Reads text directly from TXT files.

    `process_documents`: Orchestrates the extraction from multiple uploaded files

2. Text Chunking:

    `get_text_chunks`: Utilizes `RecursiveCharacterTextSplitter` from `langchain.text_splitter` to break down the extracted text into smaller, manageable chunks. This is crucial for efficient embedding and retrieval. The chunk_size is set to 10000 and chunk_overlap to 1000.

3. Vector Store Creation:

    `get_vector_store`: Takes the text chunks and uses `GoogleGenerativeAIEmbeddings` (model "models/embedding-001") to create numerical representations (embeddings) of these chunks. It then builds a `FAISS` (Facebook AI Similarity Search) vector store, which is an efficient library for similarity search, and saves it locally as "faiss_index".

4. Conversational Chain (RAG):

    `get_conversational_chain`: Defines a PromptTemplate for the LLM (`gemini-2.5-flash`). This prompt instructs the model to act as a "Technical Advisor Chatbot," answer questions based only on the provided context, and ignore product numbers starting with '9'. If the answer isn't in the context, it provides a specific fallback message. It then loads a `load_qa_chain` from LangChain.

5. User Input and Response Generation:

    `user_input`: Loads the previously saved "faiss_index" using the same embedding model. Performs a similarity_search on the loaded vector store using the user's question (and a combined history for better context awareness) to find the most relevant document chunks (k=5).
    Passes these relevant document chunks, the user's question, and the chat history to the conversational_chain to get the AI's answer.
    Includes error handling for loading the FAISS index or generating the AI response.

6. Streamlit Application (`main`):

    a. Sets up the Streamlit page configuration (`st.set_page_config`).

    b. Initializes `st.session_state.messages` to store the chat history and `st.session_state.awaiting_bot_response` to manage the UI state during response generation.

    c. Displays existing chat messages.

    d. Provides a `st.chat_input` for users to type their questions. This input is disabled while the bot is generating a response.

    e. Includes logic to append user messages to `st.session_state.messages` and trigger bot response generation using `st.rerun()` to update the UI.

    f. Displays a "Thinking..." spinner while the bot is generating a response.

    **Sidebar**:

    a. Allows users to `st.file_uploader` to upload PDF, DOCX, or TXT documents.

    b. Saves uploaded documents temporarily to a `data` directory.

    c.  A "Submit & Process Documents" button triggers the `process_documents`, `get_text_chunks`, and `get_vector_store` functions.

    d. Provides feedback (success/warning/error messages) to the user during processing.

### How to run this application?

To run this Streamlit application, follow these steps:

1. Clone the Repository:

    Open your terminal or command prompt.
    Use git clone to download the project files.

    ```
    git clone https://github.com/d3smondwong/technical_advisor_chatbot.git
    ```
2. Navigate into the cloned project directory:

    ```
    cd [your_project_directory]
    ```
3. Set up Environment Variables:

    a. Create a `.env` file in the project directory

    b. Open the newly created .env file with a text editor and add your     Google API Key:

    ```
    GOOGLE_API_KEY="YOUR_API_KEY_HERE"
    ```
    &nbsp;&nbsp;&nbsp;&nbsp;Replace "YOUR_API_KEY_HERE" with your actual Google Generative AI API key. You can obtain &nbsp;&nbsp;&nbsp;&nbsp;one from the Google AI Studio or the Google Cloud Console.

4. Install Dependencies:

    a. Ensure you are still in the project's root directory (where requirements.txt is located).

    b. Run the following command to install all necessary Python libraries:

    ```
    pip install -r requirements.txt
    ```

5. Run the Streamlit App:

    From your project's root directory in the terminal, execute:

    ```
    streamlit run app.py

    ```
    This will open the Streamlit application in your default web browser.

### How to use the application?

1. Upload Documents: In the sidebar of the Streamlit application, click "Browse files" and upload your `.pdf`, `.docx`, or `.txt` files. Alternatively, drag and drop in your files.

2. Process Documents: Click the "Submit & Process Documents" button in the sidebar. You'll see a spinner, and then a "Documents Processed Successfully" message once the FAISS index has been created.

3. Ask Questions: Once documents are processed, type your questions in the chat input box at the bottom of the main page and press Enter. The chatbot will retrieve relevant information from your uploaded documents and attempt to answer your question.
