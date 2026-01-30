# coding=utf-8
"""
FastAPI server for Qwen3-TTS VoiceDesign model.

Usage:
    uvicorn api_server:app --host 0.0.0.0 --port 8000

API Endpoints:
    POST /tts/voice_design - Generate speech with voice design
    GET /health - Health check
    GET /languages - Get supported languages
"""

import io
import os
import base64
from typing import Optional, List
from contextlib import asynccontextmanager

import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from qwen_tts import Qwen3TTSModel

# Global model instance
model: Optional[Qwen3TTSModel] = None

# Model path - can be set via environment variable QWEN3_TTS_MODEL_PATH
MODEL_PATH = os.environ.get(
    "QWEN3_TTS_MODEL_PATH",
    "/ubuntu-22.04/Qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
)


class VoiceDesignRequest(BaseModel):
    """Request body for voice design TTS."""
    text: str = Field(..., description="Text to synthesize")
    instruct: str = Field(..., description="Voice style instruction describing desired voice/style")
    language: str = Field(default="Auto", description="Language: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian, or Auto")
    max_new_tokens: int = Field(default=2048, description="Maximum number of new tokens to generate")
    temperature: float = Field(default=0.9, description="Sampling temperature")
    top_p: float = Field(default=1.0, description="Top-p sampling parameter")
    top_k: int = Field(default=50, description="Top-k sampling parameter")
    response_format: str = Field(default="wav", description="Response format: 'wav' for audio file, 'base64' for base64 encoded audio")


class VoiceDesignBatchRequest(BaseModel):
    """Request body for batch voice design TTS."""
    texts: List[str] = Field(..., description="List of texts to synthesize")
    instructs: List[str] = Field(..., description="List of voice style instructions")
    languages: List[str] = Field(default=None, description="List of languages for each text")
    max_new_tokens: int = Field(default=2048, description="Maximum number of new tokens to generate")
    temperature: float = Field(default=0.9, description="Sampling temperature")
    top_p: float = Field(default=1.0, description="Top-p sampling parameter")
    top_k: int = Field(default=50, description="Top-k sampling parameter")


class TTSResponse(BaseModel):
    """Response for base64 encoded audio."""
    audio_base64: str
    sample_rate: int
    format: str = "wav"


class BatchTTSResponse(BaseModel):
    """Response for batch TTS."""
    audios: List[str]  # base64 encoded
    sample_rate: int
    format: str = "wav"


def get_worker_device() -> str:
    """Get GPU device for current worker based on worker ID and available GPUs."""
    if not torch.cuda.is_available():
        return "cpu"
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return "cpu"
    
    # Get worker ID from environment (set by uvicorn)
    # Each worker will use a different GPU in round-robin fashion
    worker_id = os.getpid() % num_gpus
    return f"cuda:{worker_id}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model
    print(f"Loading Qwen3-TTS model from {MODEL_PATH}...")
    
    # Get device for this worker (distributes across available GPUs)
    device = get_worker_device()
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    # Try to use flash attention if available
    attn_impl = "flash_attention_2" if torch.cuda.is_available() else "eager"
    
    print(f"Worker PID {os.getpid()} using device: {device}")
    
    try:
        model = Qwen3TTSModel.from_pretrained(
            MODEL_PATH,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_impl,
        )
        print(f"Model loaded successfully on {device}")
    except Exception as e:
        print(f"Warning: Failed to load with flash_attention_2, falling back to eager: {e}")
        model = Qwen3TTSModel.from_pretrained(
            MODEL_PATH,
            device_map=device,
            dtype=dtype,
            attn_implementation="eager",
        )
        print(f"Model loaded successfully on {device} with eager attention")
    
    yield
    
    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="Qwen3-TTS API",
    description="FastAPI server for Qwen3-TTS VoiceDesign model",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "cuda_available": torch.cuda.is_available(),
    }


@app.get("/languages")
async def get_languages():
    """Get supported languages."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    languages = model.get_supported_languages()
    return {"languages": languages}


@app.post("/tts/voice_design")
async def voice_design_tts(request: VoiceDesignRequest):
    """
    Generate speech using voice design.
    
    The instruct parameter describes the desired voice characteristics,
    such as age, gender, emotion, speaking style, etc.
    
    Example instruct values:
    - "体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显"
    - "Speak in an incredulous tone, with a hint of panic"
    - "A warm, gentle young female voice"
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        wavs, sr = model.generate_voice_design(
            text=request.text,
            instruct=request.instruct,
            language=request.language,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
        )
        
        # Convert to wav bytes
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, wavs[0], sr, format="WAV")
        audio_buffer.seek(0)
        
        if request.response_format == "base64":
            audio_base64 = base64.b64encode(audio_buffer.read()).decode("utf-8")
            return TTSResponse(
                audio_base64=audio_base64,
                sample_rate=sr,
                format="wav"
            )
        else:
            return StreamingResponse(
                audio_buffer,
                media_type="audio/wav",
                headers={"Content-Disposition": "attachment; filename=output.wav"}
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts/voice_design/batch", response_model=BatchTTSResponse)
async def voice_design_batch_tts(request: VoiceDesignBatchRequest):
    """
    Batch generate speech using voice design.
    
    All texts will be processed in a single batch for efficiency.
    Returns base64 encoded audio for each input text.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(request.texts) != len(request.instructs):
        raise HTTPException(
            status_code=400, 
            detail=f"texts and instructs must have same length: {len(request.texts)} vs {len(request.instructs)}"
        )
    
    languages = request.languages
    if languages is None:
        languages = ["Auto"] * len(request.texts)
    elif len(languages) != len(request.texts):
        raise HTTPException(
            status_code=400,
            detail=f"languages must have same length as texts: {len(languages)} vs {len(request.texts)}"
        )
    
    try:
        wavs, sr = model.generate_voice_design(
            text=request.texts,
            instruct=request.instructs,
            language=languages,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
        )
        
        # Convert all to base64
        audios_base64 = []
        for wav in wavs:
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, wav, sr, format="WAV")
            audio_buffer.seek(0)
            audios_base64.append(base64.b64encode(audio_buffer.read()).decode("utf-8"))
        
        return BatchTTSResponse(
            audios=audios_base64,
            sample_rate=sr,
            format="wav"
        )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
