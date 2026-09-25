from __future__ import annotations
import argparse
import re
from pathlib import Path

from vector_store import PolicyVectorStore

def chunk_markdown(path: Path) -> list[dict]:
    text = path.read_text()
    #split on '##' headings, keeping the heading with its body
    sections = re.split(r"\n(?=##)", text)
    chunks = []
    for i, section in enumerate(sections):
        section = section.strip()
        if not section or section.startswith('#'):
            #skip the top-level "# Refund Policy" title-only fragment
            if section.startswith("#") and "##" not in section:
                continue

            chunks.append(
                {
                    "policy_id": f"{path.stem} :: chunk_{i}",
                    "text": section,
                    "source_doc": path.name,
                }
            )
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

        