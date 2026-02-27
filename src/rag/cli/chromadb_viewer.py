"""ChromaDB Viewer - A Streamlit app for browsing ChromaDB collections."""

import sys

import streamlit as st

from src.config import settings
from src.models.embedding_model import EmbeddingModel
from src.retrieval.retrievers import BM25Retriever, DenseRetriever, HybridRetriever
from src.storage.vector_store import VectorStore
from src.observability.langfuse_client import configure_langfuse

# Configure LangFuse for @observe decorators
configure_langfuse()

@st.cache_resource
def get_embedding_model() -> EmbeddingModel:
    """Get cached embedding model instance."""
    return EmbeddingModel(
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        azure_endpoint=settings.azure_openai_endpoint,
        deployment_name=settings.azure_openai_embedding_deployment_name,
    )


def get_vector_store(persist_dir: str, collection_name: str) -> VectorStore:
    """Get vector store instance."""
    return VectorStore(
        persist_directory=persist_dir,
        collection_name=collection_name,
    )


def main():
    st.set_page_config(
        page_title="ChromaDB Viewer",
        page_icon="🔍",
        layout="wide",
    )

    st.title("ChromaDB Viewer")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        persist_dir = st.text_input(
            "ChromaDB Directory",
            value=settings.chroma_persist_directory,
            help="Path to the ChromaDB persistence directory",
        )

    # Get collections using a temporary VectorStore to access the client
    try:
        temp_store = get_vector_store(persist_dir, "temp")
        collections = temp_store.client.list_collections()
    except Exception as e:
        st.error(f"Failed to connect to ChromaDB: {e}")
        st.info(f"Make sure the directory '{persist_dir}' exists and contains a valid ChromaDB database.")
        return

    if not collections:
        st.warning("No collections found in the database.")
        return

    # Collection selector
    with st.sidebar:
        collection_names = [c.name for c in collections]
        selected_collection = st.selectbox(
            "Select Collection",
            options=collection_names,
            index=0,
        )

    # Get the selected collection's vector store
    vector_store = get_vector_store(persist_dir, selected_collection)
    doc_count = vector_store.count()

    # Collection info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Collection", selected_collection)
    with col2:
        st.metric("Documents", doc_count)
    with col3:
        metadata = vector_store.collection.metadata or {}
        st.metric("Distance Metric", metadata.get("hnsw:space", "unknown"))

    st.divider()

    # Tabs for different views
    tab_browse, tab_search, tab_vectors, tab_stats = st.tabs(
        ["Browse", "Search", "Vector Space", "Statistics"]
    )

    with tab_browse:
        render_browse_tab(vector_store, doc_count)

    with tab_search:
        render_search_tab(vector_store)

    with tab_vectors:
        render_vectors_tab(vector_store, doc_count)

    with tab_stats:
        render_stats_tab(vector_store, doc_count)


def render_browse_tab(vector_store: VectorStore, doc_count: int):
    """Render the browse documents tab."""
    st.subheader("Browse Documents")

    if doc_count == 0:
        st.info("No documents in this collection.")
        return

    # View mode toggle
    col_controls1, col_controls2, col_controls3 = st.columns([1, 1, 2])
    with col_controls1:
        page_size = st.selectbox("Per page", [10, 25, 50, 100], index=0)
    with col_controls2:
        total_pages = (doc_count + page_size - 1) // page_size
        page = st.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1)
    with col_controls3:
        view_mode = st.radio("View", ["Table", "Cards"], horizontal=True)

    offset = (page - 1) * page_size

    # Fetch documents using collection directly for pagination
    results = vector_store.collection.get(
        limit=page_size,
        offset=offset,
        include=["documents", "metadatas", "embeddings"],
    )

    st.caption(f"Showing {len(results['ids'])} of {doc_count} documents (page {page}/{total_pages})")

    if view_mode == "Table":
        # Build table data
        table_data = []
        embeddings = results.get("embeddings")
        if embeddings is None:
            embeddings = []
        for i, doc_id in enumerate(results["ids"]):
            content = results["documents"][i]
            preview = content[:150] + "..." if len(content) > 150 else content
            meta = results["metadatas"][i] or {}

            # Format embedding preview
            if len(embeddings) > 0 and i < len(embeddings) and embeddings[i] is not None:
                emb = embeddings[i]
                emb_preview = f"[{', '.join(f'{v:.4f}' for v in emb[:5])}{'...' if len(emb) > 5 else ''}]"
                dims = len(emb)
            else:
                emb_preview = "N/A"
                dims = 0

            row = {
                "ID": doc_id,
                "Preview": preview.replace("\n", " "),
                "Length": len(content),
                "Dims": dims,
                "Embedding": emb_preview,
                **{k: str(v)[:50] for k, v in meta.items()},
            }
            table_data.append(row)

        st.dataframe(
            table_data,
            width="stretch",
            hide_index=True,
            column_config={
                "ID": st.column_config.TextColumn("ID", width="medium"),
                "Preview": st.column_config.TextColumn("Preview", width="large"),
                "Length": st.column_config.NumberColumn("Length", format="%d chars"),
                "Dims": st.column_config.NumberColumn("Dims", width="small"),
                "Embedding": st.column_config.TextColumn("Embedding", width="medium"),
            },
        )

        # Document detail viewer
        st.markdown("---")
        selected_id = st.selectbox(
            "View full document",
            options=results["ids"],
            index=None,
            placeholder="Select a document to view details...",
        )
        if selected_id:
            idx = results["ids"].index(selected_id)
            st.markdown(f"**Document:** `{selected_id}`")
            st.text_area(
                "Content",
                value=results["documents"][idx],
                height=300,
                disabled=True,
            )
            if results["metadatas"][idx]:
                st.markdown("**Metadata:**")
                st.json(results["metadatas"][idx])
            if len(embeddings) > 0 and idx < len(embeddings) and embeddings[idx] is not None:
                st.markdown(f"**Embedding:** {len(embeddings[idx])} dimensions")
                st.code(f"[{', '.join(f'{v:.6f}' for v in embeddings[idx][:20])}{'...' if len(embeddings[idx]) > 20 else ''}]")
    else:
        # Card view
        for i, doc_id in enumerate(results["ids"]):
            with st.expander(f"📄 {doc_id}", expanded=False):
                st.markdown("**Content:**")
                st.text(results["documents"][i][:2000] + ("..." if len(results["documents"][i]) > 2000 else ""))

                if results["metadatas"][i]:
                    st.markdown("**Metadata:**")
                    st.json(results["metadatas"][i])


def render_search_tab(vector_store: VectorStore):
    """Render the search tab with retriever selection."""
    st.subheader("Search")

    # Retriever selection
    col_retriever, col_search, col_k = st.columns([1, 2, 1])
    with col_retriever:
        retriever_type = st.selectbox(
            "Retriever",
            options=["Dense", "BM25", "Hybrid"],
            index=0,
            help="Dense: semantic similarity | BM25: keyword matching | Hybrid: combines both",
        )
    with col_search:
        query = st.text_input("Search query", placeholder="Enter a search query...")
    with col_k:
        default_top_k = {
            "Dense": settings.retrieval_dense_top_k,
            "BM25": settings.retrieval_sparse_top_k,
            "Hybrid": settings.retrieval_hybrid_final_top_k,
        }.get(retriever_type, 5)
        top_k = st.number_input(
            "Results",
            min_value=1,
            max_value=20,
            value=min(max(int(default_top_k), 1), 20),
        )

    # Show hybrid settings if selected
    if retriever_type == "Hybrid":
        col_dense_w, col_sparse_w = st.columns(2)
        with col_dense_w:
            dense_weight = st.slider(
                "Dense weight",
                min_value=0.0,
                max_value=1.0,
                value=settings.hybrid_search_dense_weight,
                step=0.1,
            )
        with col_sparse_w:
            sparse_weight = st.slider(
                "Sparse weight",
                min_value=0.0,
                max_value=1.0,
                value=settings.hybrid_search_sparse_weight,
                step=0.1,
            )

    if query:
        try:
            # Build the selected retriever
            embedding_model = get_embedding_model()

            if retriever_type == "Dense":
                retriever = DenseRetriever(
                    vector_store=vector_store,
                    embedding_model=embedding_model,
                    top_k=top_k,
                )
            elif retriever_type == "BM25":
                retriever = BM25Retriever(
                    vector_store=vector_store,
                    top_k=top_k,
                )
            else:  # Hybrid
                dense_retriever = DenseRetriever(
                    vector_store=vector_store,
                    embedding_model=embedding_model,
                    top_k=top_k,
                )
                sparse_retriever = BM25Retriever(
                    vector_store=vector_store,
                    top_k=top_k,
                )
                retriever = HybridRetriever(
                    dense_retriever=dense_retriever,
                    sparse_retriever=sparse_retriever,
                    dense_weight=dense_weight,
                    sparse_weight=sparse_weight,
                    final_top_k=top_k,
                )

            results = retriever.search(query)

            if results:
                # Results table
                search_data = []
                for i, result in enumerate(results):
                    # Score interpretation depends on retriever type
                    if retriever_type == "Dense":
                        # Dense uses cosine distance: lower = more similar
                        display_score = 1 - result.score
                        score_label = "Similarity"
                    elif retriever_type == "BM25":
                        # BM25 score: higher = more relevant
                        display_score = result.score
                        score_label = "BM25 Score"
                    else:
                        # Hybrid uses RRF score: higher = better
                        display_score = result.score
                        score_label = "RRF Score"

                    preview = result.text[:150] + "..." if len(result.text) > 150 else result.text

                    search_data.append({
                        "Rank": i + 1,
                        "Score": display_score,
                        "ID": result.doc_id,
                        "Preview": preview.replace("\n", " "),
                        "Source": result.metadata.get("source", result.metadata.get("filename", "-")),
                    })

                # Determine score range for progress bar
                max_score = max(d["Score"] for d in search_data) if search_data else 1
                min_score = 0 if retriever_type == "Dense" else min(d["Score"] for d in search_data)

                st.caption(f"Retriever: **{retriever_type}** | Score type: {score_label}")
                st.dataframe(
                    search_data,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Rank": st.column_config.NumberColumn("Rank", width="small"),
                        "Score": st.column_config.ProgressColumn(
                            score_label,
                            min_value=min_score,
                            max_value=max_score if max_score > 0 else 1,
                            format="%.3f",
                        ),
                        "ID": st.column_config.TextColumn("ID", width="medium"),
                        "Preview": st.column_config.TextColumn("Preview", width="large"),
                        "Source": st.column_config.TextColumn("Source", width="medium"),
                    },
                )

                # Detailed view
                st.markdown("---")
                for i, result in enumerate(results):
                    if retriever_type == "Dense":
                        display_score = 1 - result.score
                    else:
                        display_score = result.score

                    with st.expander(f"#{i+1} {result.doc_id} ({display_score:.3f})", expanded=i == 0):
                        st.text_area(
                            "Content",
                            value=result.text,
                            height=200,
                            disabled=True,
                            key=f"search_content_{i}",
                        )
                        if result.metadata:
                            st.json(result.metadata)
            else:
                st.info("No results found.")
        except Exception as e:
            st.error(f"Search failed: {e}")


def render_vectors_tab(vector_store: VectorStore, doc_count: int):
    """Render the vector space visualization tab."""
    st.subheader("Vector Space Visualization")

    if doc_count == 0:
        st.info("No documents to visualize.")
        return

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    col1, col2 = st.columns([2, 1])
    with col1:
        max_points = st.slider(
            "Max documents to visualize",
            min_value=10,
            max_value=min(500, doc_count),
            value=min(100, doc_count),
        )
    with col2:
        top_k = st.number_input("Search results", min_value=1, max_value=20, value=5)

    # Search input
    search_query = st.text_input(
        "Search in vector space",
        placeholder="Enter a query to see its position and nearest neighbors...",
        key="vector_search",
    )

    if st.button("Generate Visualization", type="primary"):
        with st.spinner("Fetching embeddings and computing projection..."):
            try:
                from sklearn.decomposition import PCA
                from sklearn.manifold import TSNE

                # Fetch embeddings from collection
                results = vector_store.collection.get(
                    limit=max_points,
                    include=["embeddings", "metadatas", "documents"],
                )

                embeddings = results["embeddings"]
                if embeddings is None or len(embeddings) == 0:
                    st.warning("No embeddings found in collection.")
                    return

                ids = results["ids"]
                metadatas = results["metadatas"]
                documents = results["documents"]

                embeddings_array = np.array(embeddings)

                # First reduce with PCA to 50 dims for speed
                n_components_pca = min(50, embeddings_array.shape[1], len(embeddings) - 1)
                pca = PCA(n_components=n_components_pca)
                embeddings_pca = pca.fit_transform(embeddings_array)

                # Then TSNE to 2D
                perplexity = min(30, len(embeddings) - 1)
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                embeddings_2d = tsne.fit_transform(embeddings_pca)

                # Store in session state for search
                st.session_state["vector_viz"] = {
                    "ids": ids,
                    "embeddings_array": embeddings_array,
                    "embeddings_2d": embeddings_2d,
                    "metadatas": metadatas,
                    "documents": documents,
                    "pca": pca,
                    "pca_mean": embeddings_pca.mean(axis=0),
                    "tsne_mean": embeddings_2d.mean(axis=0),
                }

                st.success("Visualization generated! You can now search.")

            except ImportError:
                st.error("Install scikit-learn for vector visualization: `uv add scikit-learn`")
                return
            except Exception as e:
                st.error(f"Visualization failed: {e}")
                return

    # Display visualization if available
    if "vector_viz" not in st.session_state:
        st.info("Click 'Generate Visualization' to compute the 2D projection.")
        return

    viz = st.session_state["vector_viz"]
    ids = viz["ids"]
    embeddings_array = viz["embeddings_array"]
    embeddings_2d = viz["embeddings_2d"]
    metadatas = viz["metadatas"]
    documents = viz["documents"]
    pca = viz["pca"]

    # Build base scatter data
    scatter_data = []
    for i in range(len(ids)):
        meta = metadatas[i] or {}
        source = meta.get("source", meta.get("filename", "unknown"))
        preview = documents[i][:100].replace("\n", " ") if documents[i] else ""
        emb = embeddings_array[i]
        emb_preview = f"[{', '.join(f'{v:.4f}' for v in emb[:5])}{'...' if len(emb) > 5 else ''}]"

        scatter_data.append({
            "x": float(embeddings_2d[i, 0]),
            "y": float(embeddings_2d[i, 1]),
            "id": ids[i],
            "source": source,
            "preview": preview,
            "embedding": emb_preview,
            "dims": len(emb),
            "type": "document",
        })

    # Handle search
    search_results = []
    query_point = None

    if search_query:
        try:
            # Get query embedding
            embedding_model = get_embedding_model()
            query_embedding = embedding_model.embed(search_query)
            query_emb_array = np.array(query_embedding).reshape(1, -1)

            # Project query to 2D using stored PCA
            query_pca = pca.transform(query_emb_array)

            # Find approximate position using nearest neighbor interpolation
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=min(5, len(embeddings_array)))
            nn.fit(pca.transform(embeddings_array))
            distances, indices = nn.kneighbors(query_pca)

            # Weighted average of nearest neighbors' 2D positions
            weights = 1 / (distances[0] + 1e-6)
            weights /= weights.sum()
            query_2d = np.average(embeddings_2d[indices[0]], axis=0, weights=weights)

            query_point = {
                "x": float(query_2d[0]),
                "y": float(query_2d[1]),
                "id": "QUERY",
                "source": "query",
                "preview": search_query[:100],
                "embedding": f"[{', '.join(f'{v:.4f}' for v in query_embedding[:5])}...]",
                "dims": len(query_embedding),
                "type": "query",
            }

            # Get search results using DenseRetriever
            retriever = DenseRetriever(
                vector_store=vector_store,
                embedding_model=embedding_model,
                top_k=top_k,
            )
            search_results = retriever.search(search_query)

            # Mark search results in scatter data
            result_ids = {r.doc_id for r in search_results}
            for item in scatter_data:
                if item["id"] in result_ids:
                    item["type"] = "search_result"

        except Exception as e:
            st.error(f"Search failed: {e}")

    # Create figure
    df = pd.DataFrame(scatter_data)
    fig = go.Figure()

    # Add documents
    docs_df = df[df["type"] == "document"]
    if not docs_df.empty:
        fig.add_trace(go.Scatter(
            x=docs_df["x"],
            y=docs_df["y"],
            mode="markers",
            marker=dict(size=8, color="#636EFA", opacity=0.6),
            name="Documents",
            text=docs_df["preview"],
            customdata=docs_df["id"],
            hovertemplate="<b>%{customdata}</b><br>%{text}<extra></extra>",
        ))

    # Add search results (highlighted)
    results_df = df[df["type"] == "search_result"]
    if not results_df.empty:
        fig.add_trace(go.Scatter(
            x=results_df["x"],
            y=results_df["y"],
            mode="markers",
            marker=dict(size=14, color="#00CC96", symbol="star", line=dict(width=1, color="white")),
            name="Search Results",
            text=results_df["preview"],
            customdata=results_df["id"],
            hovertemplate="<b>%{customdata}</b><br>%{text}<extra></extra>",
        ))

    # Add query point
    if query_point:
        fig.add_trace(go.Scatter(
            x=[query_point["x"]],
            y=[query_point["y"]],
            mode="markers+text",
            marker=dict(size=18, color="#EF553B", symbol="x", line=dict(width=2, color="white")),
            name="Query",
            text=["QUERY"],
            textposition="top center",
            hovertemplate=f"<b>Query</b><br>{search_query[:50]}<extra></extra>",
        ))

        # Draw lines from query to results
        for item in scatter_data:
            if item["type"] == "search_result":
                fig.add_trace(go.Scatter(
                    x=[query_point["x"], item["x"]],
                    y=[query_point["y"], item["y"]],
                    mode="lines",
                    line=dict(color="#EF553B", width=1, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                ))

    fig.update_layout(
        title="Document Embeddings (t-SNE projection)" + (f" - Query: '{search_query[:30]}...'" if search_query else ""),
        xaxis=dict(scaleanchor="y", scaleratio=1, title=""),
        yaxis=dict(title=""),
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show search results table if searching
    if search_results:
        st.markdown("**Search Results**")
        results_data = []
        for i, result in enumerate(search_results):
            similarity = 1 - result.score
            results_data.append({
                "Rank": i + 1,
                "Similarity": similarity,
                "ID": result.doc_id,
                "Preview": result.text[:150].replace("\n", " "),
            })
        st.dataframe(
            results_data,
            width="stretch",
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", width="small"),
                "Similarity": st.column_config.ProgressColumn("Similarity", min_value=0, max_value=1, format="%.3f"),
                "ID": st.column_config.TextColumn("ID", width="medium"),
                "Preview": st.column_config.TextColumn("Preview", width="large"),
            },
        )

    # Show all vectors table
    st.markdown("**All Document Vectors**")
    st.dataframe(
        scatter_data,
        width="stretch",
        hide_index=True,
        column_config={
            "x": st.column_config.NumberColumn("X (2D)", format="%.2f", width="small"),
            "y": st.column_config.NumberColumn("Y (2D)", format="%.2f", width="small"),
            "id": st.column_config.TextColumn("ID", width="medium"),
            "source": st.column_config.TextColumn("Source", width="small"),
            "type": st.column_config.TextColumn("Type", width="small"),
            "dims": st.column_config.NumberColumn("Dims", width="small"),
            "embedding": st.column_config.TextColumn("Embedding (preview)", width="large"),
            "preview": st.column_config.TextColumn("Content", width="medium"),
        },
    )


def render_stats_tab(vector_store: VectorStore, doc_count: int):
    """Render the statistics tab."""
    st.subheader("Collection Statistics")

    if doc_count == 0:
        st.info("No documents to analyze.")
        return

    # Use get_all_documents from VectorStore
    all_docs = vector_store.get_all_documents()

    # Document length statistics
    doc_lengths = [len(doc["text"]) for doc in all_docs]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Documents", len(doc_lengths))
    with col2:
        st.metric("Avg Length", f"{sum(doc_lengths) / len(doc_lengths):.0f} chars")
    with col3:
        st.metric("Min Length", f"{min(doc_lengths)} chars")
    with col4:
        st.metric("Max Length", f"{max(doc_lengths)} chars")

    # Document length distribution chart
    st.markdown("**Document Length Distribution**")
    st.bar_chart({"Length (chars)": doc_lengths})

    # Metadata statistics table
    all_keys = set()
    for doc in all_docs:
        if doc["metadata"]:
            all_keys.update(doc["metadata"].keys())

    if all_keys:
        st.markdown("**Metadata Fields**")
        metadata_stats = []
        for key in sorted(all_keys):
            values = [doc["metadata"].get(key) for doc in all_docs if doc["metadata"] and key in doc["metadata"]]
            unique_values = set(str(v) for v in values)
            metadata_stats.append({
                "Field": key,
                "Documents": len(values),
                "Coverage": f"{len(values) / len(all_docs) * 100:.1f}%",
                "Unique Values": len(unique_values),
                "Sample Values": ", ".join(list(unique_values)[:3]) + ("..." if len(unique_values) > 3 else ""),
            })

        st.dataframe(
            metadata_stats,
            width="stretch",
            hide_index=True,
            column_config={
                "Field": st.column_config.TextColumn("Field", width="medium"),
                "Documents": st.column_config.NumberColumn("Documents", width="small"),
                "Coverage": st.column_config.TextColumn("Coverage", width="small"),
                "Unique Values": st.column_config.NumberColumn("Unique", width="small"),
                "Sample Values": st.column_config.TextColumn("Sample Values", width="large"),
            },
        )

        # Metadata value breakdown
        st.markdown("**Metadata Value Distribution**")
        selected_field = st.selectbox("Select field", options=sorted(all_keys))
        if selected_field:
            value_counts = {}
            for doc in all_docs:
                if doc["metadata"] and selected_field in doc["metadata"]:
                    val = str(doc["metadata"][selected_field])
                    value_counts[val] = value_counts.get(val, 0) + 1

            value_data = [
                {"Value": k, "Count": v}
                for k, v in sorted(value_counts.items(), key=lambda x: -x[1])
            ]
            st.dataframe(
                value_data,
                width="stretch",
                hide_index=True,
                column_config={
                    "Count": st.column_config.ProgressColumn(
                        "Count",
                        min_value=0,
                        max_value=max(value_counts.values()),
                    ),
                },
            )
    else:
        st.info("No metadata found in documents.")


def run():
    """Entry point for the chromadb-viewer command."""
    import subprocess

    script_path = __file__
    subprocess.run([sys.executable, "-m", "streamlit", "run", script_path, "--server.headless", "true"])


if __name__ == "__main__":
    main()
