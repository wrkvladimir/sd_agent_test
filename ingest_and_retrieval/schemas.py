from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class IngestRequest(BaseModel):
    source_type: str = "file"  # "file" | "inline_html"
    source_id: str = "test_html"
    path: Optional[str] = "data/test.html"
    html: Optional[str] = None
    collection: str = "support_knowledge"
    recreate_collection: bool = False


class IngestResponse(BaseModel):
    status: str
    collection: str
    source_id: str
    articles_total: int
    paragraphs_total: int
    indexed_points: int
    warnings: List[str] = []


class Chunk(BaseModel):
    id: Any
    text: str = ""
    metadata: Dict[str, Any] = {}
    score: Optional[float] = None


class SearchDebugInfo(BaseModel):
    used_threshold: float
    initial_candidates_count: int
    after_threshold_count: int
    vector_scores: List[float] = []
    rerank_scores: List[float] = []


class SearchResponse(BaseModel):
    chunks: List[Chunk]
    debug: Optional[SearchDebugInfo] = None


class AllDataResponse(BaseModel):
    collection: str
    total_points: int
    vector_size: int | None = None
    distance: str | None = None
    points: List[Dict[str, Any]]


class ClearDataResponse(BaseModel):
    collection: str
    deleted_points: int


class SearchRequest(BaseModel):
    # Единственное обязательное поле — текст запроса.
    query: str
    # Дополнительные параметры берутся из конфигурации/окружения.
    with_debug: bool = True


class ParagraphChunk(BaseModel):
    # Единственное обязательное содержимое — текст абзаца.
    text_paragraph: str
    text_for_embedding: str

    # Всё остальное может отсутствовать в исходных данных.
    title: Optional[str] = None
    source_id: Optional[str] = None
    source_date: Optional[str] = None
    source_label: Optional[str] = None
    ingested_at: Optional[str] = None
    source_document_id: Optional[str] = None
