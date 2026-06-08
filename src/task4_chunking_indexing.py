"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# TODO: Chọn chunking strategy và giải thích vì sao
CHUNK_SIZE = 500        # 500 ký tự đủ ngắn để search chính xác, không quá dài cho LLM
CHUNK_OVERLAP = 50     # overlap 50 giúp không mất ngữ cảnh giữa 2 chunk
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# TODO: Chọn embedding model và giải thích
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # nhẹ, dễ chạy local
EMBEDDING_DIM = 384

# TODO: Chọn vector store
VECTOR_STORE = "chromadb"  # "weaviate" | "chromadb" | "faiss"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    # TODO: Iterate qua STANDARDIZED_DIR, đọc .md files
    documents = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type}
        })
    return documents
    # raise NotImplementedError("Implement load_documents")


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    # TODO: Implement chunking
    #
    # Ví dụ với RecursiveCharacterTextSplitter:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i}
            })
    return chunks
    # raise NotImplementedError("Implement chunk_documents")


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    # TODO: Implement embedding
    #
    # Ví dụ với sentence-transformers:
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks
    # raise NotImplementedError("Implement embed_chunks")


# def index_to_vectorstore(chunks: list[dict]):
#     """
#     Lưu chunks vào vector store đã chọn.
#     """
#     # TODO: Implement indexing
#     #
#     # Ví dụ với Weaviate:
#     # import weaviate
#     # from weaviate.classes.config import Configure, Property, DataType
#     #
#     # client = weaviate.connect_to_local()  # hoặc connect_to_weaviate_cloud()
#     #
#     # # Tạo collection
#     # collection = client.collections.create(
#     #     name="DrugLawDocs",
#     #     vectorizer_config=Configure.Vectorizer.none(),
#     #     properties=[
#     #         Property(name="content", data_type=DataType.TEXT),
#     #         Property(name="source", data_type=DataType.TEXT),
#     #         Property(name="doc_type", data_type=DataType.TEXT),
#     #     ]
#     # )
#     #
#     # # Insert chunks
#     # with collection.batch.dynamic() as batch:
#     #     for chunk in chunks:
#     #         batch.add_object(
#     #             properties={"content": chunk["content"], ...},
#     #             vector=chunk["embedding"]
#     #         )
#     raise NotImplementedError("Implement index_to_vectorstore")
# def index_to_vectorstore(chunks: list[dict]):
#     """
#     Lưu chunks vào Weaviate local.
#     """
#     import weaviate
#     from weaviate.classes.config import Configure, Property, DataType

#     client = weaviate.connect_to_local()

#     collection_name = "DrugLawDocs"

#     # Nếu collection đã tồn tại thì xóa để index lại từ đầu
#     if client.collections.exists(collection_name):
#         client.collections.delete(collection_name)

#     # Tạo collection mới
#     collection = client.collections.create(
#         name=collection_name,
#         vectorizer_config=Configure.Vectorizer.none(),
#         properties=[
#             Property(name="content", data_type=DataType.TEXT),
#             Property(name="source", data_type=DataType.TEXT),
#             Property(name="doc_type", data_type=DataType.TEXT),
#             Property(name="chunk_index", data_type=DataType.INT),
#         ],
#     )

#     # Insert chunks kèm vector embedding
#     with collection.batch.dynamic() as batch:
#         for chunk in chunks:
#             metadata = chunk["metadata"]

#             batch.add_object(
#                 properties={
#                     "content": chunk["content"],
#                     "source": metadata.get("source", ""),
#                     "doc_type": metadata.get("type", ""),
#                     "chunk_index": metadata.get("chunk_index", 0),
#                 },
#                 vector=chunk["embedding"],
#             )

#     failed = collection.batch.failed_objects
#     if failed:
#         print(f"  ⚠ Failed to insert {len(failed)} objects")
#     else:
#         print(f"  ✓ Saved {len(chunks)} chunks to Weaviate collection: {collection_name}")

#     client.close()

def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào ChromaDB local.
    """
    import chromadb

    index_dir = Path(__file__).parent.parent / "data" / "vectorstore"
    index_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(index_dir))

    collection_name = "drug_law_docs"

    # Xóa collection cũ nếu có để index lại từ đầu
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(name=collection_name)

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for i, chunk in enumerate(chunks):
        ids.append(f"chunk_{i}")
        documents.append(chunk["content"])
        metadatas.append(chunk["metadata"])
        embeddings.append(chunk["embedding"])

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print(f"  ✓ Saved {len(chunks)} chunks to ChromaDB")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
