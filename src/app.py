from flask import Flask, request, jsonify, render_template
from rag import ChatDocument
import os
import webbrowser
from threading import Timer
import tempfile
import atexit
import shutil

app = Flask(__name__)

# Create a temporary directory for uploads
uploads_dir = tempfile.mkdtemp(prefix="lumina_uploads_")
print(f"Uploads will be stored temporarily in: {uploads_dir}")

# Ensure the temporary directory is deleted when the app exits
def cleanup_uploads():
    if os.path.exists(uploads_dir):
        print(f"Cleaning up temporary uploads directory: {uploads_dir}")
        shutil.rmtree(uploads_dir)

atexit.register(cleanup_uploads)

# Initialize the ChatDocument instance
rag = ChatDocument(uploads_dir)

# Serve the main UI
@app.route('/')
def index():
    return render_template('index.html')

# Handle user queries
@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    query = data.get('query')
    if not query:
        return jsonify({'error': 'No query provided'}), 400

    response = rag.ask(query)
    return jsonify({'response': response})

# Handle document uploads
@app.route('/upload', methods=['POST'])
def upload():
    if 'files' not in request.files:
        return jsonify({'success': False, 'message': 'No file part'}), 400

    files = request.files.getlist('files')
    if not files:
        return jsonify({'success': False, 'message': 'No files selected'}), 400

    for file in files:
        file_path = os.path.join(uploads_dir, file.filename)
        file.save(file_path)
        try:
            rag.ingest(file_path)
        except ValueError as e:
            return jsonify({'success': False, 'message': str(e)}), 400

    return jsonify({'success': True, 'message': 'Files uploaded successfully'})

@app.route('/clear', methods=['POST'])
def clear():
    if os.path.exists(uploads_dir):
        for file in os.listdir(uploads_dir):
            file_path = os.path.join(uploads_dir, file)
            os.remove(file_path)
    rag.clear()
    return jsonify({'success': True, 'message': 'Session cleared.'})

@app.route('/agent_status', methods=['GET'])
def agent_status():
    status = "running" if rag.db_agent_thread and rag.db_agent_thread.is_alive() else "idle"
    return jsonify({'status': status})

@app.route('/agent_insights', methods=['GET'])
def agent_insights():
    insights = rag.document_summaries
    cleaned_insights = {}

    for doc_id, summary in insights.items():
        # Extract the filename from the path
        filename = os.path.basename(doc_id)
        
        # Clean up the summary text
        cleaned_summary = summary.replace(doc_id, "")  # Remove the file path
        cleaned_insights[filename] = cleaned_summary

    #print(f"Returning cleaned insights: {cleaned_insights}")
    return jsonify(cleaned_insights)

# Function to open the browser
def open_browser():
    webbrowser.open('http://127.0.0.1:5000/')

if __name__ == '__main__':
    # Open the browser after 1 second
    Timer(1, open_browser).start()

    rag.start_agent_process()
    
    # Run the Flask app
    app.run(debug=False)