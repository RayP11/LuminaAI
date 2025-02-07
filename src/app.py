from rag import ChatDocument

rag = ChatDocument()

rag.load_documents()

print("ready for input: ")
query = input()

while query.lower() != "exit":
    response = rag.ask(query)
    print(response + "\n")
    query = input()