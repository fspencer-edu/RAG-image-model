import streamlit as st

from app.ui import init_session_state, render_footer, render_main, render_sidebar

st.set_page_config(page_title="Image RAG + Captions + LLM", layout="wide")
st.title("Image RAG + Captions + LLM")
st.caption(
    "Upload images, auto-caption them, retrieve the most relevant ones, "
    "and generate an answer from the retrieved captions."
)

def main() -> None:
    init_session_state()
    render_sidebar()
    render_footer()
    render_main()
    
if __name__ == "__main__":
    main()