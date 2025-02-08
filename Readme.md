# Multilingual Chatbot

A sophisticated chatbot system that can detect languages, analyze sentiment, translate messages, and generate contextual responses across multiple languages.

## Features

- Language Detection: Automatically identifies input language using XLM-RoBERTa
- Sentiment Analysis: Analyzes the emotional tone of messages using BERT
- Translation Support: Translates between English and Romance languages (French, Spanish, Italian)
- Response Generation: Creates contextual responses using GPT-2
- Multiple Interfaces: 
  - RESTful API (FastAPI)
  - Web Interface (HTML/JavaScript)
  - Streamlit UI

## Tech Stack

- **Backend Framework**: FastAPI
- **Machine Learning**: PyTorch, Transformers
- **Frontend**: HTML, JavaScript, Streamlit
- **Models**:
  - Language Detection: papluca/xlm-roberta-base-language-detection
  - Sentiment Analysis: nlptown/bert-base-multilingual-uncased-sentiment
  - Translation: Helsinki-NLP/opus-mt models
  - Text Generation: GPT-2

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/multilingual-chatbot.git
cd multilingual-chatbot
```

2. Install dependencies:
```bash
pip install torch transformers fastapi uvicorn streamlit requests pydantic
```

## Usage

1. Start the FastAPI server:
```bash
python bot.py
```

2. Access the interfaces:
- API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Streamlit UI: Run `streamlit run bot.py`
- Web Interface: Open `chatbot.html` in a browser

## API Endpoints

### POST /chat
Send a message to the chatbot:
```json
{
    "text": "Your message here",
    "source_language": "english",
    "target_language": "english"
}
```

Response format:
```json
{
    "original_text": "Your message here",
    "detected_language": "english",
    "sentiment": {
        "sentiment": "POSITIVE",
        "score": 0.95
    },
    "response": "Generated response here"
}
```

## Project Structure

```
multilingual-chatbot/
├── bot.py              # Main application file
├── chatbot.html        # Web interface
└── README.md           # Documentation
```


## License

This project is licensed under the MIT License - see the LICENSE file for details.
