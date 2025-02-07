import torch
from transformers import pipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import streamlit as st
import requests

# Define the request model
class ChatMessage(BaseModel):
    text: str
    source_language: str
    target_language: str = "english"

# Chatbot System Class
class ChatbotSystem:
    def __init__(self):
        try:
            # Initialize language detection model
            self.language_detector = pipeline("zero-shot-classification", 
                                              model="papluca/xlm-roberta-base-language-detection")
            
            # Initialize multilingual sentiment analysis
            self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                               model="nlptown/bert-base-multilingual-uncased-sentiment")
            
            # Initialize translation models
            self.translator_en_to_romance = pipeline("translation_en_to_fr", 
                                                     model="Helsinki-NLP/opus-mt-en-ROMANCE")
            self.translator_romance_to_en = pipeline("translation_fr_to_en", 
                                                     model="Helsinki-NLP/opus-mt-ROMANCE-en")
            
            # Initialize response generation model
            self.response_model = pipeline("text-generation", 
                                           model="gpt2")
        except Exception as e:
            raise RuntimeError(f"Error initializing models: {str(e)}")
    
    def detect_language(self, text: str) -> str:
        """Detect the language of the input text."""
        result = self.language_detector(text, candidate_labels=["english", "french", "spanish", "italian"])
        return result['labels'][0]
    
    def analyze_sentiment(self, text: str) -> dict:
        """Analyze the sentiment of the input text."""
        result = self.sentiment_analyzer(text)[0]
        return { 'sentiment': result['label'], 'score': result['score'] }
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text between supported languages."""
        try:
            if source_lang == "english" and target_lang in ["french", "spanish", "italian"]:
                result = self.translator_en_to_romance(text)[0]
            elif source_lang in ["french", "spanish", "italian"] and target_lang == "english":
                result = self.translator_romance_to_en(text)[0]
            else:
                return text  # Return original text if translation is not supported
            return result['translation_text']
        except Exception:
            return text  # Fallback to original text if translation fails
    
    def generate_response(self, text: str) -> str:
        """Generate a chatbot response based on input text."""
        response = self.response_model(text, max_length=100, num_return_sequences=1)[0]
        return response['generated_text']
    
    def process_message(self, message: ChatMessage) -> dict:
        """Process an incoming message and generate an appropriate response."""
        detected_lang = self.detect_language(message.text)
        sentiment = self.analyze_sentiment(message.text)
        
        processed_text = self.translate_text(message.text, detected_lang, "english") if detected_lang != "english" else message.text
        response = self.generate_response(processed_text)
        
        if message.target_language != "english":
            response = self.translate_text(response, "english", message.target_language)
        
        return {
            "original_text": message.text,
            "detected_language": detected_lang,
            "sentiment": sentiment,
            "response": response
        }

# Initialize FastAPI app
app = FastAPI(title="Multilingual Chatbot API")
chatbot = ChatbotSystem()

@app.post("/chat")
async def chat(message: ChatMessage):
    """API endpoint for chatbot interactions."""
    try:
        return chatbot.process_message(message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Streamlit UI
st.title("Multilingual Chatbot")
user_input = st.text_input("Type your message...")
if st.button("Send"):
    response = requests.post("http://127.0.0.1:8000/chat", json={"text": user_input, "source_language": "english"})
    bot_response = response.json().get("response", "No response from the bot.")
    st.write(f"**You:** {user_input}")
    st.write(f"**Bot:** {bot_response}")

# HTML/JavaScript Frontend
html_code = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multilingual Chatbot</title>
</head>
<body>
    <h1>Multilingual Chatbot</h1>
    <form id="chat-form">
        <input type="text" id="user-input" placeholder="Type your message..." required>
        <button type="submit">Send</button>
    </form>
    <div id="chat-response"></div>

    <script>
        const form = document.getElementById('chat-form');
        const userInput = document.getElementById('user-input');
        const chatResponse = document.getElementById('chat-response');

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            const message = userInput.value;
            const response = await fetch('http://127.0.0.1:8000/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: message, source_language: "english" }),
            });
            const data = await response.json();
            chatResponse.innerHTML = `<p><strong>You:</strong> ${message}</p><p><strong>Bot:</strong> ${data.response}</p>`;
            userInput.value = '';
        });
    </script>
</body>
</html>
'''

with open("chatbot.html", "w") as file:
    file.write(html_code)

# Run the FastAPI application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)