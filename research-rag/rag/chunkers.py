from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

STRATEGIES = ["fixed", "recursive", "markdown", "semantic"]


def _fixed(docs, size, overlap):
    splitter = CharacterTextSplitter(separator="", chunk_size=size, chunk_overlap=overlap)
    return splitter.split_documents(docs)


def _recursive(docs, size, overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return splitter.split_documents(docs)


def _markdown(docs, size, overlap):
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    sections = []
    for d in docs:
        for s in header_splitter.split_text(d.page_content):
            sections.append(Document(page_content=s.page_content, metadata={**d.metadata, **s.metadata}))
    return _recursive(sections, size, overlap)  # long sections still need cutting


def _semantic(docs, embeddings):
    from langchain_experimental.text_splitter import SemanticChunker

    splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    return splitter.split_documents(docs)


def chunk_documents(docs, strategy="recursive", size=1000, overlap=150, embeddings=None):
    if strategy == "fixed":
        chunks = _fixed(docs, size, overlap)
    elif strategy == "recursive":
        chunks = _recursive(docs, size, overlap)
    elif strategy == "markdown":
        chunks = _markdown(docs, size, overlap)
    elif strategy == "semantic":
        if embeddings is None:
            raise ValueError("semantic chunking needs an embeddings object")
        chunks = _semantic(docs, embeddings)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {STRATEGIES}")
    return _add_ids(chunks, strategy)


MIN_CHUNK_CHARS = 50  # drop orphan headings / table fragments that are useless for search


def _drop_tiny(chunks):
    """Remove chunks shorter than MIN_CHUNK_CHARS, unless it is the only chunk of its page/sheet."""
    per_page = {}
    for c in chunks:
        key = (c.metadata.get("source"), c.metadata.get("page"))
        per_page[key] = per_page.get(key, 0) + 1
    return [
        c for c in chunks
        if len(c.page_content.strip()) >= MIN_CHUNK_CHARS
        or per_page[(c.metadata.get("source"), c.metadata.get("page"))] == 1
    ]


def _add_ids(chunks, strategy):
    """Give every chunk a unique id and keep only simple metadata (Chroma needs str/int/float/bool)."""
    chunks = _drop_tiny(chunks)
    counter = {}
    for c in chunks:
        key = (c.metadata.get("source"), c.metadata.get("page"))
        counter[key] = counter.get(key, 0) + 1
        c.metadata = {k: v for k, v in c.metadata.items() if isinstance(v, (str, int, float, bool))}
        c.metadata["chunk_index"] = counter[key]
        c.metadata["strategy"] = strategy
        c.id = f"{key[0]}|{key[1]}|{counter[key]}"
    return chunks
