from langchain_chroma import Chroma
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
import cv2
import pytesseract
from PIL import Image
from moviepy import VideoFileClip
import whisper


class ChatDocument:
    def __init__(self, uploads_dir):
        self.uploads_dir = uploads_dir
        self.model = ChatOllama(model="llama3.2")
        self.text_splitter = CharacterTextSplitter(separator='\n', chunk_size=2000, chunk_overlap=200)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text") 
        self.vector_store = None
        self.retriever = None
        self.chain = None

        # Initialize Chroma vector store
        self._initialize_vector_store()

        self.prompt = PromptTemplate.from_template(
            """
            You are Lumina AI. You are an AI RAG Model assistant meant for healthcare professionals.
            You will assist with analyzing patient documents, keeping track of patients' progress,
            and helping the healthcare professionals evaluate their treatment plan.
            Keep your responses brief but informative. Only answer what the user asks.
            Use your chat history for a conversative experience that is user-friendly.
            Chat History: {chat_history}
            Question: {question} 
            Context: {context} 
            Answer: 
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
        # extracts the text and audio of a video
        elif file_extension in ["mp4", "avi", "mov"]:
            text = self._analyze_video(file_path)
            from langchain.schema import Document
            docs = [Document(page_content=text)]
            chunks = self.text_splitter.split_documents(docs)
            return filter_complex_metadata(chunks)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        docs = loader.load()
        chunks = self.text_splitter.split_documents(docs)
        return filter_complex_metadata(chunks)
    
    def _extract_video_text(self, video_path: str, frame_interval: int=10):
        """Extract text from video"""
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        extracted_text = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process every x frame
            if frame_count % frame_interval == 0:
                # Covert frame to PIL image
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # Extract any text displayed in the video
                text = pytesseract.image_to_string(pil_image)
                extracted_text.append(text)
            frame_count += 1

        cap.release()
        return "\n".join(extracted_text)
    
    def _extract_video_audio(self, video_path, output_audio_path):
        """Extract the audio data from a video"""
        #get video audio
        video = VideoFileClip(video_path)
        audio = video.audio

        #write audio data into a new audio file
        audio.write_audiofile(output_audio_path, codec= 'pcm_s16le')

        video.close()
        audio.close()

    def audio_to_text(self, audio_path):
        """Utilize OpenAI Whisper model to transcribe audio data into text"""
        model = whisper.load_model("small")
        result = model.transcribe(audio_path)
        return result["text"].strip() if result["text"] else "could not understand audio"
    
    def _analyze_video(self, video_path: str):
        """Analyze video contents"""
        #get the text from the video
        frames_text = self._extract_video_text(video_path)
        #get the audio from the video
        self._extract_video_audio(video_path, "extracted_audio.wav")
        audio_text = self.audio_to_text("extracted_audio.wav")
        #combine audio and displayed text from video
        analysis_result = f"Extracted Text from Frames:{frames_text} Extracted text from audio:{audio_text}"
        return analysis_result
        

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