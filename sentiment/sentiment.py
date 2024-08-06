from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langdetect import detect
from translate import Translator
from transformers import pipeline
import emoji

app = FastAPI()

# CORS configuration
origins = [
    "http://localhost:8001",  # Frontend's port
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods including OPTIONS
    allow_headers=["*"],  # Allow all headers
)

class LanguageTranslator:
    def __init__(self, target_language='en'):
        self.target_language = target_language

    def detect_language(self, text):
        """Detect the language of the given text."""
        language = detect(text)
        return language

    def detect_language_and_translate(self, text):
        """Detect the source language and translate the text to the target language."""
        source_language = self.detect_language(text)
        translator = Translator(from_lang=source_language, to_lang=self.target_language)
        translated_text = translator.translate(text)
        return translated_text

class SentimentAnalyzer:
    def __init__(self, target_language='en', max_chunk_size=512):
        self.translator = LanguageTranslator(target_language=target_language)
        self.pipeline = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.max_chunk_size = max_chunk_size

    def preprocess_text(self, text):
        """Convert emojis to their textual descriptions."""
        return emoji.demojize(text)

    def generate_sentiment(self, text):
        """Classify the sentiment of the given text as positive, negative, or neutral."""
        translated_text = self.translator.detect_language_and_translate(text)
        text = self.preprocess_text(translated_text)
        result = self.pipeline(text)
        sentiment_label = result[0]['label']
        sentiment_score = result[0]['score']
        if sentiment_label in ['positive', 'negative'] and sentiment_score < 0.6:
            return 'Neutral'
        else:
            return sentiment_label.capitalize()

    def analyze_large_text(self, text):
        """Analyze sentiment of large text by breaking it into paragraphs and chunks, returning sentiment percentages."""
        # Split the text into paragraphs
        paragraphs = text.split('\n\n')

        sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        
        for paragraph in paragraphs:
            # Chunk the paragraph into smaller parts if necessary
            # print(paragraph)
            chunks = [paragraph[i:i + self.max_chunk_size] for i in range(0, len(paragraph), self.max_chunk_size)]
            
            for chunk in chunks:
                if chunk.strip():  # Ignore empty chunks
                    sentiment = self.generate_sentiment(chunk)
                    sentiment_counts[sentiment] += 1

        # Calculate percentages
        total_chunks = sum(sentiment_counts.values())
        sentiment_percentages = {sentiment: (count / total_chunks) * 100 for sentiment, count in sentiment_counts.items()}

        return sentiment_percentages

class TextRequest(BaseModel):
    text: str

@app.post("/analyze-sentiment/")
async def analyze_sentiment(request: TextRequest):
    try:
        sentiment_analyzer = SentimentAnalyzer()
        sentiment_percentages = sentiment_analyzer.analyze_large_text(request.text)
        return {"sentiment_percentages": sentiment_percentages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("sentiment:app", host="0.0.0.0", port=8001, reload=True)
