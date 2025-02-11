document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const uploadBtn = document.getElementById('upload-btn');
    const fileInput = document.getElementById('file-input');
    
    // Function to add a message to the chat box
    function addMessage(message, isUser) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message');
        messageElement.classList.add(isUser ? 'user-message' : 'ai-message');

        const textElement = document.createElement('div');
        textElement.classList.add('ai-text');
        // Format the AI's response
        textElement.innerHTML = formatResponse(message);

        messageElement.appendChild(textElement);
        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // Function to format the AI's response
    function formatResponse(text) {
        // Replace Markdown horizontal rules with an empty string
        text = text.replace(/(^|\n)---($|\n)/g, '');  // Remove ---
        text = text.replace(/(^|\n)\*\*\*($|\n)/g, '');  // Remove ***
        text = text.replace(/(^|\n)___($|\n)/g, '');  // Remove ___
    
        // Replace Markdown-like syntax with HTML
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>'); // Bold
        text = text.replace(/\*(.*?)\*/g, '<em>$1</em>'); // Italic
        text = text.replace(/`(.*?)`/g, '<code>$1</code>'); // Inline code
        text = text.replace(/```([\s\S]*?)```/g, '<pre>$1</pre>'); // Code block
        text = text.replace(/^> (.*$)/gm, '<blockquote>$1</blockquote>'); // Blockquote
    
        // Preserve line breaks
        text = text.replace(/\n/g, '<br>');
    
        return text;
    }

    // Function to handle sending a message
    async function sendMessage() {
        const query = userInput.value.trim();
        if (query) {
            addMessage(query, true);
            userInput.value = '';

            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ query }),
                });

                const data = await response.json();
                addMessage(data.response, false);
            } catch (error) {
                addMessage('Error: Could not get a response from the server.', false);
            }
        }
    }

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

    // Event listeners
    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    uploadBtn.addEventListener('click', uploadDocument);

    const clearBtn = document.getElementById('clear-btn');

    clearBtn.addEventListener('click', async () => {
        try {
            const response = await fetch('/clear', {
                method: 'POST',
            });

            const data = await response.json();
            if (data.success) {
                alert('Session cleared.');
                chatBox.innerHTML = ''; // Clear the chat box
            } else {
                alert('Failed to clear session.');
            }
        } catch (error) {
            alert('Error clearing session.');
        }
    });
});