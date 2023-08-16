import os
import PyPDF2
import random
import csv
from docx import Document
from io import BytesIO
import json
from bs4 import BeautifulSoup
import itertools
import streamlit as st
from io import StringIO
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.retrievers import SVMRetriever
from langchain.chains import QAGenerationChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import CallbackManager
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.callbacks import get_openai_callback
from langchain.llms import HuggingFaceHub
from langchain.document_loaders import OnlinePDFLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from email import message_from_bytes

def extract_mhtml_content(mhtml_content):
    # Parse the MHTML content
    message = message_from_bytes(mhtml_content)

    # Extract the main HTML content
    for part in message.walk():
        content_type = part.get_content_type()
        content_disposition = part.get('Content-Disposition', '')

        if 'attachment' not in content_disposition and content_type == 'text/html':
            html_content = part.get_payload(decode=True).decode('utf-8')
            # Here we use BeautifulSoup to extract text from the HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            return soup.get_text()

    return None


st.set_page_config(page_title="Chat with multiple files",page_icon=':bar_chart:')

@st.cache_data
def load_docs(files):
    st.info("`Reading files ...`")
    all_pages = []  # Store texts for each page
    page_map = {}  # Store start and end character indices for each page
    current_idx = 0
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                all_pages.append(text)
                page_map[page_num + 1] = (current_idx, current_idx + len(text))
                current_idx += len(text)
        elif file_extension == ".txt":
            stringio = StringIO(file_path.getvalue().decode("utf-8"))
            text = stringio.read()
            all_pages.append(text)
        elif file_extension == ".csv":
            content = file_path.getvalue().decode("utf-8")
            reader = csv.reader(content.splitlines())
            for row in reader:
                all_pages.append(", ".join(row))  # Convert each row to a string
        elif file_extension == ".md":
            text = file_path.getvalue().decode("utf-8")
            all_pages.append(text)
        elif file_extension == ".html":
            soup = BeautifulSoup(file_path.getvalue().decode("utf-8"), 'html.parser')
            text = soup.get_text()
            all_pages.append(text)
        elif file_extension == ".docx":
            # Use BytesIO to create a file-like object in memory
            virtual_file = BytesIO(file_path.getvalue())
            doc = Document(virtual_file)
            text = "\n".join([para.text for para in doc.paragraphs])
            all_pages.append(text)
        elif file_extension == ".mhtml":
            text = extract_mhtml_content(file_path.getvalue())
            if text:
                all_pages.append(text)
        elif file_extension == ".json":
            content = file_path.getvalue().decode("utf-8")
            data = json.loads(content)
            text = json.dumps(data, indent=4)  # Convert the JSON object to a formatted string
            all_pages.append(text)
        else:
            st.warning('Please provide txt or pdf.', icon="⚠️")
    return all_pages, page_map





@st.cache_resource
def create_retriever(_embeddings, splits, retriever_type):
    if retriever_type == "SIMILARITY SEARCH":
        try:
            vectorstore = FAISS.from_texts(splits, _embeddings)
        except (IndexError, ValueError) as e:
            st.error(f"Error creating vectorstore: {e}")
            return
        retriever = vectorstore.as_retriever(k=5)
    elif retriever_type == "SUPPORT VECTOR MACHINES":
        retriever = SVMRetriever.from_texts(splits, _embeddings)

    return retriever

@st.cache_resource
def split_texts(text, chunk_size, overlap, split_method):
    st.info("`Splitting doc ...`")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    splits = text_splitter.split_text(text)
    if not splits:
        st.error("Failed to split document")
        st.stop()
    return splits, list(range(len(splits)))  # Return splits and their indices

def find_page_for_index(idx, page_map):
    for page, (start, end) in page_map.items():
        if start <= idx < end:
            return page
    return None


def load_online_pdf(pdf_link):
    # Handle the online PDF link here, e.g., download the PDF.
    loader = OnlinePDFLoader(pdf_link)
    return loader.load()
# ...

def main():
    
    foot = f"""
    <div style="
        position: fixed;
        bottom: 0;
        right: 5px;
        padding: 0px;
        text-align: left;
    ">
        <p><a href='mailto:harshit09795@gmail.com'>Contact</a></p>
    </div>
    """


    st.markdown(foot, unsafe_allow_html=True)
    
    # # Add custom CSS
    # st.markdown(
    #     """
    #     <style>
        
    #     #MainMenu {visibility: hidden;
    #     # }
    #         footer {visibility: hidden;
    #         }
    #         .css-card {
    #             border-radius: 0px;
    #             padding: 30px 10px 10px 10px;
    #             background-color: #f8f9fa;
    #             box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    #             margin-bottom: 10px;
    #             font-family: "IBM Plex Sans", sans-serif;
    #         }
            
    #         .card-tag {
    #             border-radius: 0px;
    #             padding: 1px 5px 1px 5px;
    #             margin-bottom: 10px;
    #             position: absolute;
    #             left: 0px;
    #             top: 0px;
    #             font-size: 0.6rem;
    #             font-family: "IBM Plex Sans", sans-serif;
    #             color: white;
    #             background-color: green;
    #             }
                
    #         .css-zt5igj {left:0;
    #         }
            
    #         span.css-10trblm {margin-left:0;
    #         }
            
    #         div.css-1kyxreq {margin-top: -40px;
    #         }
            
    #     </style>
    #     """,
    #     unsafe_allow_html=True,
    # )

    st.markdown(
    """
    <style>
        /* Hide main menu and footer */
        #MainMenu, footer {
            visibility: hidden;
        }

        /* Card styling */
        .css-card {
            border-radius: 10px; /* Rounded corners */
            padding: 30px 15px 15px 15px;
            background-color: #f8f9fa;
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.12), 0 2px 4px rgba(0, 0, 0, 0.08); /* Slight increase in shadow for depth */
            margin-bottom: 20px;
            font-family: "IBM Plex Sans", sans-serif;
            transition: transform 0.3s ease, box-shadow 0.3s ease; /* Smooth transition for hover effect */
        }
        
        /* Card hover effect */
        .css-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 12px rgba(0, 0, 0, 0.15), 0 3px 6px rgba(0, 0, 0, 0.10);
        }

        /* Card tag styling */
        .card-tag {
            border-radius: 3px; /* Slight rounded corners */
            padding: 4px 8px;
            margin-bottom: 10px;
            position: absolute;
            left: 10px; /* Some spacing from the left edge */
            top: 10px; /* Some spacing from the top edge */
            font-size: 0.7rem;
            font-family: "IBM Plex Sans", sans-serif;
            color: white;
            background-color: #27ae60; /* Slightly darker green */
            transition: background-color 0.3s ease; /* Smooth transition for hover effect */
        }

        /* Card tag hover effect */
        .card-tag:hover {
            background-color: #2ecc71; /* Slightly lighter green on hover */
        }
        
        .css-zt5igj {
            left: 0;
        }
        
        span.css-10trblm {
            margin-left: 0;
        }
        
        div.css-1kyxreq {
            margin-top: -40px;
        }
    </style>
        """,
        unsafe_allow_html=True,
)

    st.sidebar.image("img/logo1.png")
   

    st.write(
    f"""
    <div style="display: flex; align-items: center; margin-left: 0;">
        <h1 style="display: inline-block;">Chat with multiple files</h1>
    </div>
    """,
    unsafe_allow_html=True,
        )
    
    st.write(
    f"""
    <div style="display: flex; align-items: center; margin-left: 0;">
        <h6 style="display: inline-block;">This application answers questions based on the content of uploaded PDF, TXT, CSV, HTML, MD, DOC, JSON documents, providing both the answer and source references.</h6>
    </div>
    """,
    unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
        .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
        .viewerBadge_text__1JaDK {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    
    st.sidebar.title("Select Models")

    
    # Initialize session state for cb if not present
    if 'cb' not in st.session_state:
        st.session_state.cb = ""

    embedding_option = st.sidebar.radio(
        "Choose Embeddings", ["OpenAI Embeddings + LLM(For best Result)", "HuggingFace Embeddings(slower) + OpenAI LLM", "HuggingFace Embeddings(slower) + LLM(free)"])

    retriever_type = st.sidebar.selectbox(
        "Choose Retriever", ["SIMILARITY SEARCH", "SUPPORT VECTOR MACHINES"])
    
    st.sidebar.markdown("### Embedding Options Descriptions")
    st.sidebar.markdown("**OpenAI Embeddings + LLM:** Utilizes embeddings from OpenAI combined with the LLM model for retrieval. Offers fast performance and high-quality results.")
    st.sidebar.markdown("**HuggingFace Embeddings(slower) + OpenAI LLM:** Combines HuggingFace embeddings with the OpenAI LLM model. HuggingFace embeddings may be slower but potentially offers more variety.")
    st.sidebar.markdown("**HuggingFace Embeddings(slower) + LLM:** Uses HuggingFace for both embeddings and retrieval using the LLM model. Might be slower than other options but can provide diverse results.")


    # Use RecursiveCharacterTextSplitter as the default and only text splitter
    splitter_type = "RecursiveCharacterTextSplitter"

    if embedding_option == "OpenAI Embeddings + LLM(For best Result)":
        if 'openai_api_key' not in st.session_state:
            openai_api_key = st.text_input(
                'Slide in your OpenAI API key! or [secure your free key here](https://platform.openai.com/account/api-keys)', value="", placeholder="Enter the OpenAI API key which begins with sk-")
            if openai_api_key:
                st.session_state.openai_api_key = openai_api_key
                os.environ["OPENAI_API_KEY"] = openai_api_key
            else:
                return
        else:
            os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key

    elif embedding_option == "HuggingFace Embeddings(slower) + OpenAI LLM":
        if 'openai_api_key' not in st.session_state:
            openai_api_key = st.text_input(
                'Slide in your OpenAI API key! or [secure your free key here](https://platform.openai.com/account/api-keys)', value="", placeholder="Enter the OpenAI API key which begins with sk-")
            if openai_api_key:
                st.session_state.openai_api_key = openai_api_key
                os.environ["OPENAI_API_KEY"] = openai_api_key
            else:
                return
        else:
            os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key

        if 'huggingface_api_key' not in st.session_state:
            huggingface_api_key = st.text_input(
                'Slide in your HuggingFace API key! or [secure your free key here](https://huggingface.co/settings/tokens)', value="", placeholder="Enter the HuggingFace API key")
            if huggingface_api_key:
                st.session_state.huggingface_api_key = huggingface_api_key
                os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_key
            else:
                return
        else:
            os.environ["HUGGINGFACE_API_KEY"] = st.session_state.huggingface_api_key

    elif embedding_option == "HuggingFace Embeddings(slower) + LLM(free)":
        if 'huggingface_api_key' not in st.session_state:
            huggingface_api_key = st.text_input(
                'Slide in your HuggingFace API key! or [secure your free key here](https://huggingface.co/settings/tokens)', value="", placeholder="Enter the HuggingFace API key")
            if huggingface_api_key:
                st.session_state.huggingface_api_key = huggingface_api_key
                os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_key
            else:
                return
        else:
            os.environ["HUGGINGFACE_API_KEY"] = st.session_state.huggingface_api_key

    # SET Embedding and LLM        
    if embedding_option == "OpenAI Embeddings + LLM(For best Result)":
        embeddings = OpenAIEmbeddings()
    elif embedding_option == "HuggingFace Embeddings(slower) + OpenAI LLM" or "HuggingFace Embeddings(slower) + LLM(free)" :
        embeddings = HuggingFaceHubEmbeddings()

    


    # Initialize the RetrievalQA chain with streaming output
    callback_handler = StreamingStdOutCallbackHandler()
    callback_manager = CallbackManager([callback_handler])

    #set LLM
    if embedding_option == "OpenAI Embeddings + LLM(For best Result)" or "HuggingFace Embeddings(slower) + OpenAI LLM":
        llm = ChatOpenAI(
        streaming=True, callback_manager=callback_manager, verbose=True, temperature=0)
    elif embedding_option ==  "HuggingFace Embeddings(slower) + LLM(free)" :
        llm = HuggingFaceHub(repo_id="google/pegasus-cnn_dailymail", model_kwargs={"temperature":0.5, "max_length":512})


    
    # Ask the user what they want to do: provide a link or upload a file
    # choice = st.radio("What would you like to do?", ["Provide an online PDF link", "Upload a PDF or TXT Document"])

    # if choice == "Provide an online PDF link":
    #     pdf_link = st.text_input("Enter an online PDF link:")

    #     if pdf_link:
    #         # Handle the online PDF link here, e.g., download the PDF.
    #         if 'last_pdf_link' not in st.session_state or st.session_state.last_pdf_link != pdf_link:
    #             st.session_state.last_pdf_link = pdf_link
    #         docs = load_online_pdf(pdf_link)
    #         st.write("Documents processed.")
    #         # Split the document into chunks
    #         splits, split_indices = split_texts(docs, chunk_size=1000, overlap=0, split_method=splitter_type)
    #         # Display the number of text chunks
    #         num_chunks = len(splits)
    #         st.write(f"Number of text chunks: {num_chunks}")
    #         retriever = create_retriever(embeddings, splits, retriever_type)
    #         qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    #         with get_openai_callback() as cb:
    #             st.write("Ready to answer questions.")
    #             # Initialize source_chunks to an empty list before checking for user_question
    #             source_chunks = []
    #             # Question and answering
    #             user_question = st.text_input("Enter your question:")
    #             if user_question:
    #                 results = qa(user_question)
    #                 # Extract answer and split it into sentences
    #                 answer = results['result'].replace('<n>', '. ')
                    
    #                 # Extract source chunks
    #                 source_chunks = [doc.page_content for doc in results['source_documents']]
                    
    #                 st.write("Answer:", answer)

    #                 # Display each source chunk and its reference page
    #                 st.info("`References`")
    #                 for source_chunk in source_chunks:
    #                     if source_chunk:
    #                         st.write(f"Reference Chunk: {source_chunk}")

    #                 st.session_state.cb = cb
    #                 print(cb)      
    # elif choice == "Upload a PDF or TXT Document":

    uploaded_files = st.file_uploader("Upload a PDF or TXT Document", type=["pdf", "txt", "csv", "mhtml", "html", "md", "docx", "json"], accept_multiple_files=True)

    if uploaded_files:
        # Check if last_uploaded_files is not in session_state or if uploaded_files are different from last_uploaded_files
        if 'last_uploaded_files' not in st.session_state or st.session_state.last_uploaded_files != uploaded_files:
            st.session_state.last_uploaded_files = uploaded_files
        # Load and process the uploaded PDF or TXT files.
        all_pages, page_map = load_docs(uploaded_files)
        st.write("Documents uploaded and processed.")

        # Split the document into chunks
        splits, split_indices = split_texts(all_pages, chunk_size=1000, overlap=0, split_method=splitter_type)
        # Display the number of text chunks
        num_chunks = len(splits)
        st.write(f"Number of text chunks: {num_chunks}")
        retriever = create_retriever(embeddings, splits, retriever_type)
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

        with get_openai_callback() as cb:
                # Embed using OpenAI embeddings 
            st.write("Ready to answer questions.")
            # Initialize source_chunks to an empty list before checking for user_question
            source_chunks = []
            # Question and answering
            user_question = st.text_input("Enter your question:")
            if user_question:
                results = qa(user_question)
                # Extract answer and split it into sentences
                answer = results['result'].replace('<n>', '. ')
                
                # Extract source chunks
                source_chunks = [doc.page_content for doc in results['source_documents']]
                
                st.write("Answer:", answer)

                # Display each source chunk and its reference page
                st.info("`References`")
                for source_chunk in source_chunks:
                    if source_chunk:
                        st.write(f"Reference Chunk: {source_chunk}")

                        # Find the page that contains the source_chunk
                        for idx, page_text in enumerate(all_pages):  # Note: all_pages is now a list of page texts
                            if source_chunk in page_text:
                                st.write(f"Reference: Page NO. {idx + 1}")
                                st.write("")
                                st.write("")
                                break

                st.session_state.cb = cb
                # print(cb)


if __name__ == "__main__":
    main()
