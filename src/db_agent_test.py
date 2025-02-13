from rag import ChatDocument
import time

# Initialize the ChatDocument class
chat_doc = ChatDocument(uploads_dir="C:\Documents\LuminaAI\documents")

# Ingest documents
chat_doc.ingest("C:\\Documents\\LuminaAI\\documents\\Example Treatment Plan.pdf")
chat_doc.ingest("C:\\Documents\\LuminaAI\\documents\\Patient Treatment Plan.pdf")


# Start database agent processing
chat_doc.start_agent_process()

# Give agent time to process
time.sleep(60) 
chat_doc.stop_agent_process()

# Access summaries
# for doc_id, summary in chat_doc.document_summaries.items():
#     print(f"Document: {doc_id}")
#     print(summary)
#     print("-" * 40)