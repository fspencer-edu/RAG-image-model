from pathlib import Path

from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BlipForConditionalGeneration,
    BlipProcessor,
)


BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
HF_CACHE = MODELS_DIR / "huggingface"
ST_CACHE = MODELS_DIR / "sentence_transformers"

EMBEDDER_MODEL_NAME = "clip-ViT-B-32"
CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-base"
GENERATOR_MODEL_NAME = "google/flan-t5-base"


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    ST_CACHE.mkdir(parents=True, exist_ok=True)

    print("Downloading sentence-transformer embedder...")
    SentenceTransformer(
        EMBEDDER_MODEL_NAME,
        cache_folder=str(ST_CACHE),
    )

    print("Downloading BLIP captioner...")
    BlipProcessor.from_pretrained(
        CAPTION_MODEL_NAME,
        cache_dir=str(HF_CACHE),
    )
    BlipForConditionalGeneration.from_pretrained(
        CAPTION_MODEL_NAME,
        cache_dir=str(HF_CACHE),
    )

    print("Downloading FLAN-T5 generator...")
    AutoTokenizer.from_pretrained(
        GENERATOR_MODEL_NAME,
        cache_dir=str(HF_CACHE),
    )
    AutoModelForSeq2SeqLM.from_pretrained(
        GENERATOR_MODEL_NAME,
        cache_dir=str(HF_CACHE),
    )

    print(f"Done. Models cached in: {MODELS_DIR}")


if __name__ == "__main__":
    main()
