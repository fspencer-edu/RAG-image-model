# RAG Model

- Streamlit application

- Stores images as vectors
- CLIP (model)
    - Maps images and text into the same space
- FAISS (search engine)
    - Store all images as vectors

- Cosine similarity
    - Normalized vectors
    - Closer vectors have more similarities


## Application

1. Upload image
2. Embed each image as vector
3. Store in FAISS index
4. Use text to search to retrieve image matches




## Dependecies

```python
pip install streamlit sentence-transformers faiss-cpu pillow torch torchvision
```