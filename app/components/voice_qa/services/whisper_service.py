import librosa
import torch
from typing import List, Dict
import subprocess
import os
import tempfile

from app.core.whisper_loader import WhisperLoader
from app.core.utils import normalize_sinhala
from app.core.gemini_client import GeminiClient

# Reuse the project's embedding + LLM helpers
from app.components.document_processing.services.embedding_service import (
    generate_text_embedding as _generate_text_embedding,
)
from app.shared.ai.gemini_client import gemini_generate
from app.core.database import engine
from sqlalchemy import text

from jiwer import wer, cer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

# Hybrid retrieval helpers (lexical + dense + cross-encoder rerank)
from app.components.voice_qa.services.hybrid_retrieval import (
    retrieve_top_k,
    dense_retrieval,
)

class VoiceService:
    
    _embedding_tokenizer = None
    _embedding_model = None

    @staticmethod
    def transcribe_audio(file_path: str):
        processor, model, device = WhisperLoader.load()
        
        # Method 1: Try using soundfile directly first
        try:
            import soundfile as sf
            audio, sr = sf.read(file_path)
            # Resample if needed
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
        except:
            # Method 2: Use FFmpeg directly to convert to WAV
            temp_converted = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    temp_converted = tmp.name

                cmd = [
                    'ffmpeg', '-i', file_path, 
                    '-ar', '16000',  # Sample rate
                    '-ac', '1',       # Mono channel
                    '-c:a', 'pcm_s16le',  # PCM 16-bit
                    '-y',  # Overwrite output file
                    temp_converted
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                
                # Now load the converted file
                import soundfile as sf
                audio, sr = sf.read(temp_converted)
                
                # Clean up
                if temp_converted and os.path.exists(temp_converted):
                    os.remove(temp_converted)
            except Exception as e:
                if temp_converted and os.path.exists(temp_converted):
                    os.remove(temp_converted)
                # Method 3: Fall back to librosa with explicit backend
                import librosa
                import audioread
                # Try to force librosa to use ffmpeg
                audio, sr = librosa.load(file_path, sr=16000, res_type='kaiser_fast')
        
        # Process the audio file with WhisperProcessor
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)
        
        # Match model precision: FP16 only when CUDA is available.
        if device == "cuda":
            inputs = {k: v.half() for k, v in inputs.items()}
        
        with torch.no_grad():
            ids = model.generate(inputs["input_features"])
        
        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
        return text


    @staticmethod
    def standardize_southern_sinhala(text: str, context_hints: str = ""):
        normalized = normalize_sinhala(text)

        prompt = f"""
    ### INSTRUCTION
    You are a specialized Sinhala linguistic utility. Your ONLY task is to take a noisy, phonetically incorrect transcription and output the clean, standard literary Sinhala version (ලිඛිත සිංහල).

    ### CONTEXT VOCABULARY
    {context_hints}

    ### RULES
    1. **NO EXPLANATIONS:** Do not say "Here is the correction" or "Analysis". Do not provide bullet points.
    2. **STRICT OUTPUT:** Return ONLY the corrected Sinhala string. No other text.
    3. **FUZZY MATCHING:** If the transcription sounds like a name in the CONTEXT VOCABULARY, use the correct name (e.g., if 'ලංම කරන' sounds like 'ලම්බකර්ණ', use 'ලම්බකර්ණ').
    4. **MAINTAIN INTENT:** Do not change the user's question, just fix the grammar and spelling.

    ### EXAMPLES
    Input: මකේ නම මොකක්ද
    Output: මගේ නම කුමක්ද?

    Input: මෙහේ සදහන්මන ලංම කරන වන්ෂයට ඇය ත්රජුවරුම් කවුරුන්ද?
    Output: මෙහි සඳහන් වන ලම්බකර්ණ වංශයට අයත් රජවරුන් කවුරුන්ද?

    ### TARGET INPUT
    {normalized}

    ### FINAL CORRECTED OUTPUT:
    """

        gemini = GeminiClient()
        response = gemini.generate_content(prompt)
        
        # Use split or strip to ensure no trailing garbage text
        result = (response.get("text") or "").strip()
        
        # Safety: if the model still outputs a paragraph, take the last line or clean it
        if "\n" in result:
            # Sometimes models repeat the prompt; we just want the final line
            result = result.split("\n")[-1].replace("Output:", "").strip()

        return normalized, result

    @staticmethod
    def _load_embedding_model():
        if VoiceService._embedding_model is None:
            from transformers import AutoTokenizer, AutoModel

            VoiceService._embedding_tokenizer = AutoTokenizer.from_pretrained(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            VoiceService._embedding_model = AutoModel.from_pretrained(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            
    @staticmethod
    def _get_embedding(text: str):
        VoiceService._load_embedding_model()

        inputs = VoiceService._embedding_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            outputs = VoiceService._embedding_model(**inputs)

        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.numpy()
    
    
    @staticmethod
    def evaluate_audio(audio_path: str, reference_text: str):

        # 1️⃣ Transcribe using your existing Whisper logic
        predicted = VoiceService.transcribe_audio(audio_path)

        # 2️⃣ Normalize both texts
        # reference_text = normalize_sinhala(reference_text)
        # predicted = normalize_sinhala(predicted)

        # 3️⃣ Compute WER & CER
        word_error = wer(reference_text, predicted)
        char_error = cer(reference_text, predicted)

        # 4️⃣ Semantic similarity
        pred_emb = VoiceService._get_embedding(predicted)
        ref_emb = VoiceService._get_embedding(reference_text)

        semantic_score = cosine_similarity(pred_emb, ref_emb)[0][0]

        return {
            "prediction": predicted,
            "reference": reference_text,
            "wer": round(word_error, 4),
            "cer": round(char_error, 4),
            "semantic_similarity": round(float(semantic_score), 4)
        }

class VoiceQAService:
    """Collection of helpers for the voice → RAG → LLM pipeline.

    These functions were moved here so `voice_router` can remain small and
    simply call high-level helpers.
    """

    @staticmethod
    def generate_text_embedding(text: str) -> List[float]:
        return _generate_text_embedding(text)

    @staticmethod
    def find_similar_chunks(question_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """Dense-only retrieval wrapper.

        This method preserves the original signature (takes an embedding) but
        delegates to the safe, parameterized `dense_retrieval` implemented in
        `hybrid_retrieval.py`.
        """

        if not question_embedding:
            return []

        # delegate to hybrid.dense_retrieval which uses parameterized queries
        return dense_retrieval(question_embedding, top_k=top_k)

    @staticmethod
    def build_prompt(question: str, chunks: List[Dict]) -> str:
        system = (
            "Use the following retrieved educational content to answer the question. "
            "Answer only using the provided context. Respond in Sinhala."
        )

        chunk_texts = "\n\n".join([f"{i+1}. {c.get('text','')}" for i, c in enumerate(chunks)])

        user = f"Question: \"{question}\"\n\nRelevant content:\n{chunk_texts}"

        prompt = f"SYSTEM:\n{system}\n\nUSER:\n{user}"
        return prompt

    @staticmethod
    def llm_generate(prompt: str) -> str:
        try:
            return gemini_generate(prompt)
        except Exception as e:
            print(f"[VoiceQAService] LLM generation failed: {e}")
            return ""
        
