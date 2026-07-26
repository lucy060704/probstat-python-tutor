"""Data contracts for the future local RAG course knowledge base."""

from probstat_tutor.rag.loader import (
    RagSourceLoadError,
    RagSourceLoadErrorCode,
    load_rag_source,
    validate_resolved_source_path,
)
from probstat_tutor.rag.schemas import (
    AllowedUsage,
    AnswerLeakageRisk,
    KnowledgeSource,
    RagManifest,
    RagManifestError,
    SourceLanguage,
    SourceLicense,
    SourceType,
    load_rag_manifest,
)
from probstat_tutor.rag.source_schemas import (
    ChunkingEligibility,
    EligibilityRejectionCode,
    FormulaExplanation,
    LoadedRagSource,
    MisconceptionExplanation,
    PythonConnection,
    RagSourceDocument,
    assess_chunking_eligibility,
)

__all__ = [
    "AllowedUsage",
    "AnswerLeakageRisk",
    "ChunkingEligibility",
    "EligibilityRejectionCode",
    "FormulaExplanation",
    "KnowledgeSource",
    "LoadedRagSource",
    "MisconceptionExplanation",
    "PythonConnection",
    "RagManifest",
    "RagManifestError",
    "RagSourceDocument",
    "RagSourceLoadError",
    "RagSourceLoadErrorCode",
    "SourceLanguage",
    "SourceLicense",
    "SourceType",
    "assess_chunking_eligibility",
    "load_rag_manifest",
    "load_rag_source",
    "validate_resolved_source_path",
]
