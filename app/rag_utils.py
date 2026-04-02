from typing import Any

import faiss
import numpy as np
import torch
from PIL import Image

from app.models import load_captioner, load_embedder, load_generator


def image_to_pil(uploaded_file) -> Image.Image:
    return Image.open(uploaded_file).convert("RGB")


def normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return vectors / norms


def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def generate_caption(image: Image.Image) -> str:
    processor, model = load_captioner()
    device = _get_device()

    model = model.to(device)
    model.eval()

    image = image.convert("RGB")

    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output_ids = model.generate(**inputs, max_new_tokens=50)

    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()


def build_index(images: list[Image.Image]) -> tuple[faiss.IndexFlatIP, list[str]]:
    if not images:
        raise ValueError("images list cannot be empty")

    embedder = load_embedder()

    captions = []
    for img in images:
        caption = generate_caption(img)
        if not caption:
            caption = "An image with unclear visual content."
        captions.append(caption)

    embeddings = embedder.encode(
        captions,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    embeddings = normalize_embeddings(embeddings.astype(np.float32))

    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(embeddings, dtype=np.float32))

    return index, captions


def search_images(index: faiss.IndexFlatIP, query: str, top_k: int = 3):
    embedder = load_embedder()

    query_vector = embedder.encode(
        [str(query)],
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    query_vector = normalize_embeddings(query_vector.astype(np.float32))
    query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)

    scores, indices = index.search(query_vector, top_k)
    return indices[0], scores[0]


def generate_rag_answer(query: str, retrieved_records: list[dict[str, Any]]) -> str:
    tokenizer, model = load_generator()
    device = _get_device()

    safe_query = str(query).strip()

    context_lines = []
    for i, record in enumerate(retrieved_records, start=1):
        caption = str(record.get("caption", "")).strip()
        if caption:
            context_lines.append(f"Image {i}: {caption}")

    context = "\n".join(context_lines) if context_lines else "No captions available."

    prompt = f"""
You are answering questions about a small image collection.
Use only the provided retrieved image captions.
If the answer is unclear, say that the retrieved captions do not provide enough detail.

User question:
{safe_query}

Retrieved captions:
{context}

Write a short answer in 2-4 sentences.
""".strip()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=120)

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()