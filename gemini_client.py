import os
import google.generativeai as genai
from dotenv import load_dotenv

class GeminiClient:
    """
    Client for interacting with Gemma 3 27b model via Google Gemini API
    """
    
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Get API key from environment
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables. Please add it to your .env file.")
        
        # Configure the Gemini API
        genai.configure(api_key=api_key)
        
        # Use Gemma 3 27b model
        self.model_name = "gemma-3-27b-it"
        try:
            self.model = genai.GenerativeModel(self.model_name)
            print(f"Successfully initialized Gemma 3 27b model via Gemini API")
        except Exception as e:
            print(f"Error initializing model: {e}")
            print("Note: Make sure your API key has access to the Gemma 3 27b model")
            raise
    
    def query(self, prompt, history=None):
        """
        Send a query to the Gemma 3 27b model via Google Gemini API
        
        Args:
            prompt (str): The prompt to send to the model
            history (list, optional): List of conversation messages in the format
                [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            
        Returns:
            str: The model's response
        """
        try:
            if history:
                # Convert history to Gemini's format
                gemini_history = []
                for msg in history:
                    role = msg["role"]
                    # Convert "assistant" role to "model" for Gemini API
                    if role == "assistant":
                        role = "model"
                    gemini_history.append({"role": role, "parts": [msg["content"]]})
                print(gemini_history)
                # Start chat with history and send current prompt
                chat = self.model.start_chat(history=gemini_history)
                response = chat.send_message(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        top_p=0.9,
                        top_k=40,
                        max_output_tokens=1000,
                    )
                )
            else:
                # Simple generate_content for single prompts without history
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        top_p=0.9,
                        top_k=40,
                        max_output_tokens=1000,
                    )
                )
            
            return response.text
            
        except Exception as e:
            return f"Error generating response with Gemma 3 27b: {str(e)}"

    def stream_query(self, prompt, history=None):
        """Send a streaming query to the Gemma 3 27b model."""
        try:
            if history:
                gemini_history = []
                for msg in history:
                    role = msg["role"]
                    if role == "assistant":
                        role = "model"
                    gemini_history.append({"role": role, "parts": [msg["content"]]})
                
                chat = self.model.start_chat(history=gemini_history)
                response_stream = chat.send_message(
                    prompt,
                    stream=True,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        top_p=0.9,
                        top_k=40,
                        max_output_tokens=1000,
                    )
                )
            else:
                response_stream = self.model.generate_content(
                    prompt,
                    stream=True,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        top_p=0.9,
                        top_k=40,
                        max_output_tokens=1000,
                    )
                )
            
            for chunk in response_stream:
                try:
                    yield chunk.text
                except ValueError:
                    # This can happen with the last chunk of the stream if it's empty.
                    # We can safely ignore it.
                    continue

        except Exception as e:
            yield f"Error during streaming: {str(e)}"
    
    def get_model_info(self):
        """
        Get information about the current model
        
        Returns:
            dict: Model information
        """
        return {
            "model_name": self.model_name,
            "provider": "Google Gemini API",
            "type": "Gemma 3 27b Instruction Tuned"
        } 