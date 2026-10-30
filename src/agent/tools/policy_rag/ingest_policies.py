from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.tools.policy_rag.vector_store import PolicyVectorStore


def chunk_markdown(path: Path) -> list[dict]:
    """
    Splits policy markdown files into discrete, coherent chunks based on '##' headers.
    Skips the root '# Document Title' header so each chunk is a distinct, self-contained rule.
    """
    text = path.read_text(encoding="utf-8")
    sections = re.split(r"\n(?=##)", text)
    chunks = []
    chunk_index = 1
    for section in sections:
        section = section.strip()
        if not section:
            continue
        # Skip top-level title header (e.g. '# Refund Policy') if it has no subsection content
        if section.startswith("#") and not section.startswith("##"):
            continue

        chunks.append(
            {
                "policy_id": f"{path.stem}::chunk_{chunk_index}",
                "text": section,
                "source_doc": path.name,
            }
        )
        chunk_index += 1
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies-dir", default="policies")
    parser.add_argument("--out", default="policy_index.pkl")
    args = parser.parse_args()

    policies_dir = Path(args.policies_dir)
    all_chunks = []
    for md_file in sorted(policies_dir.glob("*.md")):
        file_chunks = chunk_markdown(md_file)
        all_chunks.extend(file_chunks)
        print(f"{md_file.name}: {len(file_chunks)} chunks")

    store = PolicyVectorStore()
    store.build(all_chunks)
    store.save(args.out)
    print(f"Indexed {len(all_chunks)} chunks total -> {args.out}")


if __name__ == "__main__":
    main()
