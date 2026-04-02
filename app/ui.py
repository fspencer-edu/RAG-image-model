import streamlit as st

from app.rag_utils import (
    build_index,
    generate_caption,
    generate_rag_answer,
    image_to_pil,
    search_images,
)


def init_session_state() -> None:
    if "image_records" not in st.session_state:
        st.session_state.image_records = []

    if "index" not in st.session_state:
        st.session_state.index = None


def render_sidebar() -> None:
    with st.sidebar:
        st.header("1) Upload images")
        uploaded_files = st.file_uploader(
            "Choose image files",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
        )

        if st.button("Build index + captions", use_container_width=True):
            if not uploaded_files:
                st.warning("Upload at least one image first.")
            else:
                image_records = []
                pil_images = []

                with st.spinner("Generating captions and embeddings..."):
                    for uploaded_file in uploaded_files:
                        image = image_to_pil(uploaded_file)
                        caption = generate_caption(image)
                        image_records.append(
                            {
                                "name": uploaded_file.name,
                                "image": image,
                                "caption": caption,
                            }
                        )
                        pil_images.append(image)

                    index = build_index(pil_images)
                    st.session_state.image_records = image_records
                    st.session_state.index = index

                st.success(
                    f"Indexed {len(image_records)} images and generated captions."
                )

        if st.button("Clear", use_container_width=True):
            st.session_state.image_records = []
            st.session_state.index = None
            st.success("Cleared stored images, captions, and index.")


def render_main() -> None:
    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("Indexed images")
        if st.session_state.image_records:
            for record in st.session_state.image_records:
                st.image(
                    record["image"],
                    caption=record["name"],
                    use_container_width=True,
                )
                st.markdown(f"**Caption:** {record['caption']}")
                st.markdown("---")
        else:
            st.info("No images indexed yet.")

    with right:
        st.subheader("2) Ask a question")
        query = st.text_input(
            "Question",
            placeholder=(
                "Which image shows a dog in a park? "
                "What image looks like a sunset scene?"
            ),
        )
        top_k = st.slider(
            "Number of retrieved images",
            min_value=1,
            max_value=5,
            value=3,
        )

        if st.button("Retrieve + Generate Answer", use_container_width=True):
            if not query.strip():
                st.warning("Enter a question first.")
            elif st.session_state.index is None:
                st.warning("Build the index first.")
            else:
                with st.spinner("Retrieving relevant images..."):
                    result_indices, result_scores = search_images(
                        st.session_state.index,
                        query,
                        top_k=top_k,
                    )

                retrieved_records = []
                st.markdown("### Retrieved images")
                for rank, (idx, score) in enumerate(
                    zip(result_indices, result_scores),
                    start=1,
                ):
                    if idx == -1:
                        continue
                    record = st.session_state.image_records[idx]
                    retrieved_records.append(record)
                    st.markdown(
                        f"**{rank}. {record['name']}** — similarity: `{score:.3f}`"
                    )
                    st.image(record["image"], use_container_width=True)
                    st.markdown(f"**Caption:** {record['caption']}")
                    st.markdown("---")

                if retrieved_records:
                    with st.spinner("Generating answer from retrieved captions..."):
                        answer = generate_rag_answer(query, retrieved_records)
                    st.markdown("### Final answer")
                    st.write(answer)
                else:
                    st.info("No relevant images found.")


def render_footer() -> None:
    st.markdown("---")
    st.subheader("How this version works")
    st.write(
        "Each uploaded image is first captioned with a vision-language model. "
        "The app also embeds each image into a shared image-text vector space "
        "using CLIP. When you ask a question, the app retrieves the most relevant "
        "images, collects their captions, and sends those captions to a language "
        "model to generate a natural-language answer."
    )
