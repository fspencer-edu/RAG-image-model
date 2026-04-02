import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BlipForConditionalGeneration,
    BlipProcessor,
)

from app.config import (
    CAPTION_MODEL_NAME,
    EMBEDDER_MODEL_NAME,
    GENERATOR_MODEL_NAME,
    MODELS_DIR,
)


@st.cache_resource
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(
        EMBEDDER_MODEL_NAME,
        cache_folder=str(MODELS_DIR / "sentence_transformers"),
    )


@st.cache_resource
def load_captioner():
    processor = BlipProcessor.from_pretrained(
        CAPTION_MODEL_NAME,
        cache_dir=str(MODELS_DIR / "huggingface"),
        local_files_only=True,
    )
    model = BlipForConditionalGeneration.from_pretrained(
        CAPTION_MODEL_NAME,
        cache_dir=str(MODELS_DIR / "huggingface"),
        local_files_only=True,
    )
    return processor, model


@st.cache_resource
def load_generator():
    tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-base",
        cache_dir="./models/huggingface",
        local_files_only=True,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-base",
        cache_dir="./models/huggingface",
        local_files_only=True,
    )
    return tokenizer, model