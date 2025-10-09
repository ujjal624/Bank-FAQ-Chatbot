// DOM Elements
const chatContainer = document.getElementById('chatContainer');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const clearBtn = document.getElementById('clearBtn');
const loading = document.getElementById('loading');

// State
let isProcessing = false;
let chatHistory = []; // Client-side history

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
    chatHistory.push({ role: 'user', content: query });
    
    const botMessageContentElement = addMessage('', 'bot');
    botMessageContentElement.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
    chatContainer.scrollTop = chatContainer.scrollHeight;

    setLoading(true);

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query, history: chatHistory })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = '';
        let isFirstChunk = true;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            if (isFirstChunk) {
                botMessageContentElement.innerHTML = '';
                isFirstChunk = false;
            }
            
            const chunk = decoder.decode(value, { stream: true });
            fullResponse += chunk;
            botMessageContentElement.innerHTML = fullResponse.replace(/\n/g, '<br>');
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        chatHistory.push({ role: 'assistant', content: fullResponse });

    } catch (error) {
        console.error('Error:', error);
        botMessageContentElement.innerHTML = 'Sorry, I encountered an error. Please try again.';
    } finally {
        setLoading(false);
    }
}

// Handle clear chat
function handleClear() {
    if (confirm('Are you sure you want to clear the chat history?')) {
        chatHistory = []; // Clear local history
        chatContainer.innerHTML = `
            <div class="welcome-message">
                <div class="welcome-icon">💬</div>
                <h2>Welcome to the FAQ Assistant</h2>
                <p>I'm here to help with your questions about our services.</p>
            </div>
        `;
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

// Set loading state (only handles button state now)
function setLoading(state) {
    isProcessing = state;
    sendBtn.disabled = state;
    
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

