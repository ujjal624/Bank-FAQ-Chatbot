// DOM Elements
const chatContainer = document.getElementById('chatContainer');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const clearBtn = document.getElementById('clearBtn');
const loading = document.getElementById('loading');

// State
let isProcessing = false;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Focus input on load
    userInput.focus();
    
    // Setup event listeners
    sendBtn.addEventListener('click', handleSend);
    clearBtn.addEventListener('click', handleClear);
    
    // Handle Enter key (Shift+Enter for new line)
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });
    
    // Auto-resize textarea
    userInput.addEventListener('input', () => {
        userInput.style.height = 'auto';
        userInput.style.height = userInput.scrollHeight + 'px';
    });
});

// Handle send message
async function handleSend() {
    const query = userInput.value.trim();
    if (!query || isProcessing) return;

    userInput.value = '';
    userInput.style.height = 'auto';

    const welcomeMessage = document.querySelector('.welcome-message');
    if (welcomeMessage) welcomeMessage.remove();

    addMessage(query, 'user');
    
    // Create a new bot message container and get a reference to its content element
    const botMessageContentElement = addMessage('', 'bot');

    setLoading(true);

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query })
        });

        setLoading(false);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value, { stream: true });
            fullResponse += chunk;
            botMessageContentElement.innerHTML = fullResponse.replace(/\n/g, '<br>');
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

    } catch (error) {
        console.error('Error:', error);
        setLoading(false);
        botMessageContentElement.innerHTML = 'Sorry, I encountered an error. Please try again.';
    }
}

// Handle clear chat
async function handleClear() {
    if (confirm('Are you sure you want to clear the chat history?')) {
        try {
            await fetch('/clear_history', {
                method: 'POST'
            });
            
            // Clear chat container
            chatContainer.innerHTML = `
                <div class="welcome-message">
                    <div class="welcome-icon">💬</div>
                    <h2>Welcome to the FAQ Assistant</h2>
                    <p>I'm here to help with your questions about our services.</p>
                </div>
            `;
            
        } catch (error) {
            console.error('Error clearing history:', error);
            alert('Failed to clear history. Please try again.');
        }
    }
}

// Add message to chat
function addMessage(content, type) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'message-avatar';
    avatarDiv.textContent = type === 'user' ? '👤' : '🤖';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = content.replace(/\n/g, '<br>');
    
    messageDiv.appendChild(avatarDiv);
    messageDiv.appendChild(contentDiv);
    
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    // Return the content element so it can be updated during streaming
    return contentDiv;
}

// Set loading state
function setLoading(state) {
    isProcessing = state;
    sendBtn.disabled = state;
    loading.style.display = state ? 'flex' : 'none';
    
    if (!state) {
        userInput.focus();
    }
}

// Format timestamp
function formatTimestamp() {
    const now = new Date();
    return now.toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit' 
    });
}

// Utility function to escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

