from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

EMBEDDER_MODEL_NAME = "clip-ViT-B-32"
CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-base"
GENERATOR_MODEL_NAME = "google/flan-t5-base"