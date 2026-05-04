import os

# Path to the saved model
MODEL_PATH = os.getenv("WHISPER_MODEL_PATH", "app/models/whisper-small-sinhala-robust-FINAL")

class WhisperLoader:
    _processor = None
    _model = None
    _device = None

    @classmethod
    def load(cls):
        # Check if model and processor are already loaded
        if cls._processor is None or cls._model is None:
            import torch
            from transformers import WhisperProcessor, WhisperForConditionalGeneration

            cls._device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load the Whisper processor and model
            cls._processor = WhisperProcessor.from_pretrained(MODEL_PATH)
            cls._model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)

            cls._model.to(cls._device)

            # FP16 is useful on GPU, but CPU Azure containers should stay FP32.
            if cls._device == "cuda":
                cls._model.half()

            cls._model.eval()
            
        return cls._processor, cls._model, cls._device
