from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import re
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel, pipeline
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda
import json
import uvicorn
from datetime import datetime
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Ticket Classification & Summarization API",
    description="BERT-based ticket classification for SAP modules and request types with text summarization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Pydantic models for request/response
class TicketRequest(BaseModel):
    subject: str = Field(..., description="Ticket subject", example="Correction in report ZFIR14")
    content: str = Field(..., description="Ticket content", example="As per our discussion on MST yesterday, please correct...")

class PredictionResult(BaseModel):
    label: str
    confidence: str

class FinalPrediction(BaseModel):
    module: str
    request_type: str
    summery: str  # Keep the typo as in original code

class TicketResponse(BaseModel):
    subject: str
    content: str
    top_module_predictions: List[PredictionResult]
    top_request_predictions: List[PredictionResult]
    final_prediction: FinalPrediction

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    summarizer_loaded: bool

# Global variables for models
model = None
tokenizer = None
summarizer = None
module_labels = ['SF', 'SAC', 'HCM', 'FICO', 'QM', 'PP', 'BASIS', 'MM', 'SD', 'ABAP']
request_labels = ['Incident', 'Service Request', 'Project', 'CR Modification', 'CR Chargeable', 'TR Movement']

# Label mappings
module_to_id = {label: idx for idx, label in enumerate(module_labels)}
id_to_module = {idx: label for label, idx in module_to_id.items()}
request_to_id = {label: idx for idx, label in enumerate(request_labels)}
id_to_request = {idx: label for label, idx in request_to_id.items()}

def clean_text(text: str) -> str:
    """Clean and preprocess text"""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def build_and_train_model():
    """Build and train the BERT model"""
    global model, tokenizer
    
    try:
        # Load data
        df = pd.read_csv("ticket_data.csv")
        df['text'] = df['subject'].fillna('') + ' ' + df['content'].fillna('')
        df['text'] = df['text'].apply(clean_text)
        df['module_label_enc'] = df['module_label'].map(module_to_id)
        df['request_type_enc'] = df['request_type'].map(request_to_id)
        df.dropna(subset=['module_label_enc', 'request_type_enc'], inplace=True)
        
        # Tokenize
        def tokenize_texts(texts, max_len=128):
            return tokenizer(list(texts), padding='max_length', truncation=True, max_length=max_len, return_tensors='np')
        
        X = tokenize_texts(df['text'])
        input_ids = X['input_ids']
        attention_mask = X['attention_mask']
        y_module = np.array(df['module_label_enc'])
        y_request = np.array(df['request_type_enc'])
        
        # Train-Test Split
        input_ids_train, input_ids_val, attn_train, attn_val, y_mod_train, y_mod_val, y_req_train, y_req_val = train_test_split(
            input_ids, attention_mask, y_module, y_request, test_size=0.2, random_state=42
        )
        
        train_inputs = {
            'input_ids': input_ids_train,
            'attention_mask': attn_train
        }
        val_inputs = {
            'input_ids': input_ids_val,
            'attention_mask': attn_val
        }
        
        # Build Model
        bert_model = TFBertModel.from_pretrained('bert-base-uncased')
        
        input_ids_layer = Input(shape=(128,), dtype=tf.int32, name='input_ids')
        attn_layer = Input(shape=(128,), dtype=tf.int32, name='attention_mask')
        
        def get_bert_output(inputs):
            input_ids, attn = inputs
            outputs = bert_model(input_ids=input_ids, attention_mask=attn)
            return outputs.pooler_output
        
        bert_output = Lambda(
            get_bert_output,
            output_shape=(768,),
            name='bert_lambda'
        )([input_ids_layer, attn_layer])
        dropout = Dropout(0.3)(bert_output)
        
        module_output = Dense(len(module_labels), activation='softmax', name='module_output')(dropout)
        request_output = Dense(len(request_labels), activation='softmax', name='request_output')(dropout)
        
        model = Model(inputs=[input_ids_layer, attn_layer], outputs=[module_output, request_output])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
            loss='sparse_categorical_crossentropy',
            metrics=[['accuracy'], ['accuracy']]
        )
        
        # Train Model
        logger.info("Training model...")
        model.fit(
            train_inputs,
            {'module_output': y_mod_train, 'request_output': y_req_train},
            validation_data=(val_inputs, {'module_output': y_mod_val, 'request_output': y_req_val}),
            epochs=3,
            batch_size=16,
            verbose=1
        )
        
        # Save model
        model.save("bert_model.h5")
        tokenizer.save_pretrained("tokenizer/")
        
        logger.info("Model training and saving completed!")
        return True
        
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        return False

def load_saved_model():
    """Load pre-trained model if available"""
    global model, tokenizer
    
    try:
        if os.path.exists("bert_model.h5") and os.path.exists("tokenizer/"):
            logger.info("Loading saved model...")
            
            # Load tokenizer
            tokenizer = BertTokenizer.from_pretrained("tokenizer/")
            
            # Load model with custom objects
            def get_bert_output(inputs):
                bert_model = TFBertModel.from_pretrained('bert-base-uncased')
                input_ids, attn = inputs
                outputs = bert_model(input_ids=input_ids, attention_mask=attn)
                return outputs.pooler_output
            
            custom_objects = {'get_bert_output': get_bert_output}
            model = load_model("bert_model.h5", custom_objects=custom_objects)
            
            logger.info("Model loaded successfully!")
            return True
    except Exception as e:
        logger.warning(f"Could not load saved model: {str(e)}")
    
    return False

def initialize_models():
    """Initialize all models"""
    global tokenizer, summarizer
    
    try:
        logger.info("Initializing models...")
        
        # Load tokenizer
        if tokenizer is None:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Load summarizer
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        
        # Try to load saved model first, otherwise train new one
        if not load_saved_model():
            if os.path.exists("ticket_data.csv"):
                logger.info("Training new model...")
                build_and_train_model()
            else:
                logger.warning("No training data found. Creating dummy model for demo.")
                # Create minimal model for demo
                bert_model = TFBertModel.from_pretrained('bert-base-uncased')
                
                input_ids_layer = Input(shape=(128,), dtype=tf.int32, name='input_ids')
                attn_layer = Input(shape=(128,), dtype=tf.int32, name='attention_mask')
                
                def get_bert_output(inputs):
                    input_ids, attn = inputs
                    outputs = bert_model(input_ids=input_ids, attention_mask=attn)
                    return outputs.pooler_output
                
                bert_output = Lambda(
                    get_bert_output,
                    output_shape=(768,),
                    name='bert_lambda'
                )([input_ids_layer, attn_layer])
                dropout = Dropout(0.3)(bert_output)
                
                module_output = Dense(len(module_labels), activation='softmax', name='module_output')(dropout)
                request_output = Dense(len(request_labels), activation='softmax', name='request_output')(dropout)
                
                model = Model(inputs=[input_ids_layer, attn_layer], outputs=[module_output, request_output])
                
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                    loss='sparse_categorical_crossentropy',
                    metrics=[['accuracy'], ['accuracy']]
                )
        
        logger.info("Models initialized successfully!")
        
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        raise e

def predict_ticket(subject: str, content: str, top_n: int = 3) -> dict:
    """Predict ticket classification and generate summary"""
    text = clean_text(subject + ' ' + content)
    encoded = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors='np')

    pred_mod, pred_req = model.predict({
        'input_ids': encoded['input_ids'], 
        'attention_mask': encoded['attention_mask']
    }, verbose=0)

    top_mod_idx = np.argsort(pred_mod[0])[::-1][:top_n]
    top_req_idx = np.argsort(pred_req[0])[::-1][:top_n]

    module_preds = [(id_to_module[i], float(pred_mod[0][i])) for i in top_mod_idx]
    request_preds = [(id_to_request[i], float(pred_req[0][i])) for i in top_req_idx]

    # Generate NLP-based summary
    input_text = (subject + " " + content).strip()
    try:
        summary_result = summarizer(input_text[:1024], max_length=60, min_length=15, do_sample=False)
        summary_text = summary_result[0]['summary_text']
    except Exception as e:
        logger.warning(f"Summarization failed: {e}. Using fallback.")
        summary_text = input_text[:100] + "..." if len(input_text) > 100 else input_text

    summary = {
        "subject": subject,
        "content": content[:200] + ("..." if len(content) > 200 else ""),
        "top_module_predictions": [
            {"label": label, "confidence": f"{conf*100:.2f}%"} for label, conf in module_preds
        ],
        "top_request_predictions": [
            {"label": label, "confidence": f"{conf*100:.2f}%"} for label, conf in request_preds
        ],
        "final_prediction": {
            "module": module_preds[0][0],
            "request_type": request_preds[0][0],
            "summery": summary_text  # Keep the typo as in original
        }
    }

    return summary

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Starting up AI Ticket Classification API...")
    initialize_models()
    logger.info("API startup completed!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Ticket Classification & Summarization API",
        "status": "active",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None and summarizer is not None else "unhealthy",
        timestamp=datetime.utcnow().isoformat(),
        model_loaded=model is not None,
        summarizer_loaded=summarizer is not None
    )

@app.post("/predict", response_model=TicketResponse)
async def predict_endpoint(request: TicketRequest):
    """
    Predict ticket classification and generate summary
    
    Example payload:
    ```json
    {
        "subject": "Correction in report ZFIR14",
        "content": "As per our discussion on MST yesterday, please correct..."
    }
    ```
    """
    if model is None or tokenizer is None or summarizer is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Please check /health endpoint.")
    
    try:
        result = predict_ticket(request.subject, request.content)
        
        return TicketResponse(
            subject=result["subject"],
            content=result["content"],
            top_module_predictions=[
                PredictionResult(label=pred["label"], confidence=pred["confidence"])
                for pred in result["top_module_predictions"]
            ],
            top_request_predictions=[
                PredictionResult(label=pred["label"], confidence=pred["confidence"])
                for pred in result["top_request_predictions"]
            ],
            final_prediction=FinalPrediction(
                module=result["final_prediction"]["module"],
                request_type=result["final_prediction"]["request_type"],
                summery=result["final_prediction"]["summery"]
            )
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/labels")
async def get_labels():
    """Get available labels"""
    return {
        "module_labels": module_labels,
        "request_labels": request_labels,
        "total_modules": len(module_labels),
        "total_request_types": len(request_labels)
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )