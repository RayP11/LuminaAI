document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const uploadBtn = document.getElementById('upload-btn');
    const fileInput = document.getElementById('file-input');
    const viewInsightsBtn = document.getElementById('view-insights-btn');
    const insightsSection = document.getElementById('insights-section');
    const chatSection = document.getElementById('chat-section');
    const insightsList = document.getElementById('insights-list');

    // Toggle between Insights and Chat sections
    viewInsightsBtn.addEventListener('click', async () => {
        const isInsightsVisible = insightsSection.style.display === 'block';

        if (isInsightsVisible) {
            // Hide insights and show chat
            insightsSection.style.display = 'none';
            chatSection.style.display = 'block';
            viewInsightsBtn.textContent = 'View Document Insights';
        } else {
            // Show insights and hide chat
            insightsSection.style.display = 'block';
            chatSection.style.display = 'none';
            viewInsightsBtn.textContent = 'Back to Chat';
            await fetchInsights(); // Fetch insights when switching to insights view
        }
    });

    async function fetchInsights() {
        console.log("Fetching insights...");  // Debugging
        try {
            const response = await fetch('/agent_insights');
            const data = await response.json();
            console.log("Insights data:", data);  // Debugging
    
            insightsList.innerHTML = '';
    
            Object.entries(data).forEach(([filename, summary]) => {
                const li = document.createElement('li');
                li.innerHTML = `
                    <div class="insight-item">
                        <h4>${filename}</h4>
                        <p>${formatResponse(summary)}</p>
                    </div>
                `;
                insightsList.appendChild(li);
            });
        } catch (error) {
            console.error("Error fetching insights:", error);  // Debugging
        }
    }

    // Function to add a message to the chat box
    function addMessage(message, isUser) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message');
        messageElement.classList.add(isUser ? 'user-message' : 'ai-message');

        const textElement = document.createElement('div');
        textElement.classList.add('ai-text');
        textElement.innerHTML = formatResponse(message);

        messageElement.appendChild(textElement);
        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // Function to format the AI's response (for better readability)
    function formatResponse(text) {
        text = text.replace(/(^|\n)---($|\n)/g, '');  // Remove ---
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>'); // Bold
        text = text.replace(/\*(.*?)\*/g, '<em>$1</em>'); // Italic
        text = text.replace(/`(.*?)`/g, '<code>$1</code>'); // Inline code
        text = text.replace(/```([\s\S]*?)```/g, '<pre>$1</pre>'); // Code block
        text = text.replace(/^> (.*$)/gm, '<blockquote>$1</blockquote>'); // Blockquote
        text = text.replace(/\n/g, '<br>'); // Preserve line breaks
        return text;
    }

    async function sendMessage() {
        const query = userInput.value.trim();
        if (!query) return;

        addMessage(query, true);
        //loading message
        addMessage("Thinking...", false);

        userInput.value = '';
        sendBtn.disabled = true;

        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query }),
            });

            const data = await response.json();
            chatBox.lastChild.remove();
            addMessage(data.response, false);
        } catch (error) {
            addMessage(`Error: ${error.message}`, false);
        } finally {
            sendBtn.disabled = false;
        }
    }

    // Event listeners for chat interactions
    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    // Function to handle document upload
    async function uploadDocument() {
        const files = fileInput.files;
        if (files.length === 0) {
            alert('Please select a file to upload.');
            return;
        }

        const formData = new FormData();
        for (const file of files) {
            formData.append('files', file);
        }

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();
            if (data.success) {
                alert('Document uploaded successfully!');
            } else {
                alert('Failed to upload document.');
            }
        } catch (error) {
            alert('Error uploading document.');
        }
    }

    // Event listener for document upload button
    uploadBtn.addEventListener('click', uploadDocument);
});
