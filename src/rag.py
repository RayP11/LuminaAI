from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, CSVLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
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
import threading
import time
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime


class ChatDocument:
    def __init__(self, uploads_dir):
        self.uploads_dir = uploads_dir
        self.model = ChatOllama(model="llama3.2")
        self.db_agent = ChatOllama(model="llama3.2")
        self.text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " "],  # Split by paragraphs, lines, and spaces
            chunk_size=2000,
            chunk_overlap=200,
        )
        self.db_agent_splitter = CharacterTextSplitter(separator='\n', chunk_size=500, chunk_overlap=100)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.db_agent_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text") 
        self.vector_store = None
        self.retriever = None
        self.db_agent_retriever = None
        self.chain = None
        self.db_agent_chain = None
        self.db_agent_thread = None
        self.db_agent_prompt = None
        self.stop_db_agent = False
        self.document_summaries = {}
        self.document_chunks = {}
        self.db_agent_queue = Queue()
        self.db_agent_instruct = """
            You are the database agent for Lumina AI, an advanced assistant for healthcare professionals. Your task is to analyze and summarize documents, identify connections between them, and provide actionable insights. Follow these steps:

            - Your responses should be short and easy to read, but informative.
            - Only provide imformation relevant to healthcare professionals looking at this document.
            - Your responses regarding the documents should give the healthcare professionals information regarding further suggestions, grouping information together, and making information easiy accessed.
            Your responses will be dislayed on a user interface so keep them in strict format.
            Don't say phrases like "the document".
            YOUR RESPONSES SHOULD BE FORMATTED PROFESSIONALLY AND NOT ROBOTIC. KEEP THEM EXTREMELY BRIEF. EMPHASIS ON BRIEF.
        """

        # Initialize Chroma vector store
        self._initialize_vector_store()

        self.prompt = PromptTemplate.from_template(
            """
            You are Lumina AI. You are an AI RAG Model assistant meant for healthcare professionals.
            You will assist with analyzing patient documents, keeping track of patients,
            and helping the healthcare professionals evaluate their treatment plan and/or notes/work.
            Keep your responses brief but informative. Only answer what the user asks.
            Use your chat history for a conversative experience that is user-friendly.
            Chat History: {chat_history}
            Question: {question} 
            Context: {context} 
            Answer: 
            """
        )

        self.db_agent_prompt = PromptTemplate.from_template(
            """
            <s> [INST]
            {prompt}
            Chat History: {chat_history}
            Current Message: {command}
            [/INST] </s>
            """
        )

    def _initialize_vector_store(self):
        """Initialize or reinitialize the Chroma vector store."""
        # Close the existing vector store if it exists
        if self.vector_store:
            self.vector_store = None
        # Clear existing vector store directory if it exists
        if os.path.exists("chroma_db"):
            shutil.rmtree("chroma_db")

        # Create a new Chroma vector store
        self.vector_store = Chroma(persist_directory="chroma_db", embedding_function=self.embeddings)
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 100,
                "score_threshold": 0.25,
            },
        )
        self.db_agent_retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 100, 
                "score_threshold": 0.1,
            },
        )


    def _load_and_split_documents(self, file_path: str):
        """Helper method to load and split documents based on file type."""
        file_extension = file_path.split(".")[-1].lower()
        docs = []

        try:
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
            elif file_extension in ["mp4", "avi", "mov"]:
                text = self._analyze_video(file_path)
                from langchain.schema import Document
                docs = [Document(page_content=text, metadata={"source": file_path, "type": "video", "timestamp": datetime.now().isoformat()})]
                chunks = self.text_splitter.split_documents(docs)
                return filter_complex_metadata(chunks)
            else:
                print(f"Unsupported file format: {file_extension}. Skipping file: {file_path}")
                return []

            docs = loader.load()
            for doc in docs:
                doc.metadata["type"] = file_extension
                doc.metadata["timestamp"] = datetime.now().isoformat()
            chunks = self.text_splitter.split_documents(docs)
            return filter_complex_metadata(chunks)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return []
    
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

        # Group chunks by document source
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            if source not in self.document_chunks:
                self.document_chunks[source] = []
            self.document_chunks[source].append(chunk)

        # Add each chunk to the background queue for processing
        for chunk in chunks:
            self.db_agent_queue.put(chunk)
            print(f"Added chunk to queue: {chunk.metadata.get('source', 'unknown')}")

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 100,
                "score_threshold": 0.25,
            },
        )

        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough(), "chat_history": self.memory.load_memory_variables}
            | self.prompt
            | self.model
            | StrOutputParser()
        )
        self.db_agent_chain = (
            {"prompt": RunnablePassthrough(), "command": RunnablePassthrough(), "chat_history": self.db_agent_memory.load_memory_variables}
            | self.db_agent_prompt
            | self.db_agent
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
    
    def start_agent_process(self):
        """activate the data base agent to process all documents."""
        if self.db_agent_thread and self.db_agent_thread.is_alive():
            print("Background processing is already running.")
            return

        self.stop_db_agent = False
        self.db_agent_thread = threading.Thread(target=self._agent_process)
        self.db_agent_thread.start()
        print("database agent activated.")

    def stop_agent_process(self):
        """Stop the database agent."""
        if self.db_agent_thread and self.db_agent_thread.is_alive():
            self.stop_db_agent = True
            self.db_agent_thread.join()
            print("agent processing stopped.")
        else:
            print("agent is no longer processing")

    def _agent_process(self):
        """Process all documents to the database agent."""
        with ThreadPoolExecutor(max_workers=2) as executor:
            while not self.stop_db_agent:
                if not self.db_agent_queue.empty():
                    doc = self.db_agent_queue.get()
                    executor.submit(self._process_document, doc)
                    self.db_agent_queue.task_done()
                else:
                    time.sleep(1)
    
    def _process_document(self, doc):
        """Process a single document and generate insight."""
        doc_id = doc.metadata.get("source", "unknown")  # Use document source as ID
        print(f"Processing document: {doc_id}")

        # Check if all chunks for this document have been processed
        if doc_id in self.document_chunks:
            # Aggregate all chunks for the document
            full_text = " ".join([chunk.page_content for chunk in self.document_chunks[doc_id]])
            
            # Summarize the full document
            summary = self._summarize_document(full_text)

            # Store the summary
            self.document_summaries[doc_id] = summary

            # store the summary in ChromaDB
            self._store_summary_in_chromadb(doc_id, summary)

            # Remove the document from the chunks dictionary to avoid reprocessing
            del self.document_chunks[doc_id]

            print(f"Summary for {doc_id}: {summary}")
            

    def _store_summary_in_chromadb(self, doc_id, summary):
        """Store the document summary in ChromaDB."""
        if self.vector_store:
            # Create a new document with the summary
            from langchain.schema import Document
            summary_doc = Document(page_content=summary, metadata={"source": doc_id, "type": "summary"})

            # Add the summary document to ChromaDB
            self.vector_store.add_documents([summary_doc])
            print(f"Stored summary in ChromaDB for document: {doc_id}")

    def _summarize_document(self, text):
        # Summarize individual chunks
        chunk_summaries = []
        for chunk in self.text_splitter.split_text(text):
            chunk_summary = self.db_agent_chain.invoke({
                "prompt": self.db_agent_instruct,
                "chat_history": self.db_agent_memory.load_memory_variables({}),
                "command": f"Summarize the following text in 1 sentence: {chunk}",
            })
            chunk_summaries.append(chunk_summary)

        # Aggregate chunk summaries into a final summary
        aggregated_summary = " ".join(chunk_summaries)
        final_summary = self.db_agent_chain.invoke({
            "prompt": self.db_agent_instruct,
            "chat_history": self.db_agent_memory.load_memory_variables({}),
            "command": f"Summarize the following text in 1-2 sentences with your educated input: {aggregated_summary}",
        })
        return final_summary

    def clear(self):
        """Clear the current state."""
        self.vector_store = None
        self.retriever = None
        self.chain = None
        self.memory.clear()
        print("State cleared.")