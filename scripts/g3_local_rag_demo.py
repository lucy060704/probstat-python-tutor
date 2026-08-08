"""Demonstrate the G3.1 local RAG path without a model or network."""

from __future__ import annotations

import argparse
import json

from probstat_tutor.config import PROJECT_ROOT
from probstat_tutor.rag import RagQuery, build_local_rag_index
from probstat_tutor.schemas import ConceptId


def main() -> None:
    parser = argparse.ArgumentParser(description="离线检索团队原创概率统计知识卡")
    parser.add_argument("query", help="要检索的统计或 Python 问题")
    parser.add_argument(
        "--concept",
        choices=[concept.value for concept in ConceptId],
        help="可选：限定到一个课程知识点",
    )
    parser.add_argument("--hint-level", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--top-k", type=int, choices=range(1, 6), default=3)
    args = parser.parse_args()

    index = build_local_rag_index(PROJECT_ROOT)
    result = index.search(
        RagQuery(
            text=args.query,
            concept_id=ConceptId(args.concept) if args.concept else None,
            disclosure_level=args.hint_level,
            top_k=args.top_k,
        )
    )
    print(result.model_dump_json(indent=2))
    print(
        json.dumps(
            {
                "indexed_sources": len(index.source_ids),
                "indexed_chunks": len(index.chunks),
                "index_fingerprint": index.index_fingerprint,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
