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
    
    if (!query || isProcessing) {
        return;
    }
    
    // Clear input and reset height
    userInput.value = '';
    userInput.style.height = 'auto';
    
    // Remove welcome message if present
    const welcomeMessage = document.querySelector('.welcome-message');
    if (welcomeMessage) {
        welcomeMessage.remove();
    }
    
    // Add user message to chat
    addMessage(query, 'user');
    
    // Show loading
    setLoading(true);
    
    try {
        // Send request to backend
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: query })
        });
        
        if (!response.ok) {
            throw new Error('Failed to get response');
        }
        
        const data = await response.json();
        
        // Hide loading
        setLoading(false);
        
        // Add bot response to chat
        addMessage(data.response, 'bot', {
            relevant: data.relevant,
            needsClarification: data.needs_clarification,
            similarityScore: data.top_similarity
        });
        
    } catch (error) {
        console.error('Error:', error);
        setLoading(false);
        addMessage('Sorry, I encountered an error processing your request. Please try again.', 'bot', { error: true });
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
                    <h2>Welcome to ICICI Bank FAQ Assistant</h2>
                    <p>I'm here to help you with questions about:</p>
                    <ul class="feature-list">
                        <li>Internet Banking & Account Services</li>
                        <li>Credit & Debit Cards</li>
                        <li>Fund Transfers & Payments</li>
                        <li>Bill Payments & Transactions</li>
                        <li>User ID & Password Management</li>
                    </ul>
                    <p class="welcome-footer">Type your question below to get started!</p>
                </div>
            `;
            
        } catch (error) {
            console.error('Error clearing history:', error);
            alert('Failed to clear history. Please try again.');
        }
    }
}

// Add message to chat
function addMessage(content, type, metadata = {}) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'message-avatar';
    avatarDiv.textContent = type === 'user' ? '👤' : '🤖';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // Format message content (preserve line breaks)
    const formattedContent = content.replace(/\n/g, '<br>');
    contentDiv.innerHTML = formattedContent;
    
    // Add metadata tags if applicable
    if (type === 'bot') {
        const metadataDiv = document.createElement('div');
        metadataDiv.className = 'message-metadata';
        
        if (metadata.relevant === false) {
            contentDiv.innerHTML += '<span class="tag tag-irrelevant">Off-topic</span>';
        }
        
        if (metadata.needsClarification) {
            contentDiv.innerHTML += '<span class="tag tag-clarification">Needs Clarification</span>';
        }
        
        // User requested to remove time and relevance score from the UI
        
        if (metadataDiv.textContent) {
            contentDiv.appendChild(metadataDiv);
        }
    }
    
    messageDiv.appendChild(avatarDiv);
    messageDiv.appendChild(contentDiv);
    
    chatContainer.appendChild(messageDiv);
    
    // Scroll to bottom
    chatContainer.scrollTop = chatContainer.scrollHeight;
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

