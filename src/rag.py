from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, CSVLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores.utils import filter_complex_metadata
import os
from contextlib import redirect_stdout, redirect_stderr
import io
import shutil


class ChatDocument:
    def __init__(self, uploads_dir):
        self.uploads_dir = uploads_dir
        self.model = ChatOllama(model="llama3.2")
        self.text_splitter = CharacterTextSplitter(separator='\n', chunk_size=2000, chunk_overlap=200)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")  # Use a single embedding model
        self.vector_store = None
        self.retriever = None
        self.chain = None

        # Initialize Chroma vector store
        self._initialize_vector_store()

        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] 
            You are Lumina AI. You are an AI RAG assistant meant for college students.
            Your primary goal is to assist students with whatever they need regarding their documents.
            Keep your responses brief but informative. Only answer what the user asks.
            Use your chat history for a conversative experience that is user-friendly.
            All of the attached files are my own files, which I've given to you.
            [/INST] </s> 
            [INST]
            Chat History: {chat_history}
            Question: {question} 
            Context: {context} 
            Answer: 
            [/INST]
            """
        )

    def _initialize_vector_store(self):
        """Initialize or reinitialize the Chroma vector store."""
        # Clear existing vector store if it exists
        if os.path.exists("chroma_db"):
            shutil.rmtree("chroma_db")

        # Create a new Chroma vector store
        self.vector_store = Chroma(persist_directory="chroma_db", embedding_function=self.embeddings)
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 100,
                "score_threshold": 0.45,
            },
        )

    def _load_and_split_documents(self, file_path: str):
        """Helper method to load and split documents based on file type."""
        file_extension = file_path.split(".")[-1].lower()
        
        if file_extension == "pdf":
            loader = PyMuPDFLoader(file_path)
        elif file_extension == "txt":
            loader = TextLoader(file_path)
        elif file_extension == "csv":
            loader = CSVLoader(file_path)
        elif file_extension == "docx":
            loader = Docx2txtLoader(file_path)
        elif file_extension == "pptx":
            loader = UnstructuredPowerPointLoader(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        docs = loader.load()
        chunks = self.text_splitter.split_documents(docs)
        return filter_complex_metadata(chunks)

    def ingest(self, file_path: str):
        """Ingest a single document."""
        if not os.path.isfile(file_path):
            raise ValueError("File path must point to a valid document.")

        chunks = self._load_and_split_documents(file_path)
        if self.vector_store:
            # Append to existing vector store
            self.vector_store.add_documents(chunks)
        else:
            # Create new vector store
            self.vector_store = Chroma.from_documents(documents=chunks, embedding=self.embeddings, persist_directory="chroma_db")

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )

        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough(), "chat_history": self.memory.load_memory_variables}
            | self.prompt
            | self.model
            | StrOutputParser()
        )
        print(f"Ingested document: {file_path}")

    def ask(self, query: str):
        """Ask a question based on the ingested documents."""
        if not self.chain:
            return "Please add a document first."

        # Redirect stdout and stderr to suppress unwanted output
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.memory.chat_memory.add_user_message(query)
            ai_response = self.chain.invoke(query)
            self.memory.chat_memory.add_ai_message(ai_response)

        return ai_response.strip()

    def clear(self):
        """Clear the current state."""
        self.vector_store = None
        self.retriever = None
        self.chain = None
        self.memory.clear()
        print("State cleared.")