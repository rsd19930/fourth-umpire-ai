import os
import re
import json
import urllib.request
from PyPDF2 import PdfReader

PDF_URL = "https://www.lords.org/getmedia/1d908298-5c44-468d-b6a7-e1414a1296e0/Laws-of-Cricket-2017-Code-4th-Edition-(2026)_3.pdf"
PDF_PATH = "laws_of_cricket.pdf"
OUTPUT_PATH = "real_cricket_rules.json"

# Page 0: cover, Pages 1-2: preface + history, Page 3: TOC (skip),
# Pages 4-83: preamble + laws + appendices, Pages 84-89: index/notes/copyright (skip)
PREFACE_PAGES = (1, 2)
CONTENT_PAGE_START = 4
CONTENT_PAGE_END = 83


def download_pdf():
    if os.path.exists(PDF_PATH):
        print(f"PDF already exists at {PDF_PATH}, skipping download.")
        return
    print("Downloading MCC Laws of Cricket PDF...")
    urllib.request.urlretrieve(PDF_URL, PDF_PATH)
    print("Download complete.")


def clean_text(text):
    """Clean a text string: normalize whitespace, remove PDF artifacts."""
    # Normalize non-breaking spaces
    text = text.replace("\xa0", " ")
    # Replace newlines with spaces (PDF line breaks are not meaningful)
    text = re.sub(r"\n+", " ", text)
    # Collapse multiple spaces into one
    text = re.sub(r"  +", " ", text)
    return text.strip()


def extract_pages(reader, start, end):
    """Extract and join text from a range of PDF pages, stripping page numbers."""
    page_texts = []
    for i in range(start, end + 1):
        text = reader.pages[i].extract_text() or ""
        # Strip leading page number pairs (e.g., "8 9", "148 149")
        text = re.sub(r"^\d{1,3}\s+\d{1,3}", "", text, count=1)
        page_texts.append(text)
    raw = "\n".join(page_texts)
    raw = raw.replace("\xa0", " ")
    return raw


def extract_preface_and_history(reader):
    """Extract preface and significant dates as separate chunks."""
    raw = extract_pages(reader, *PREFACE_PAGES)

    chunks = []

    # Split at "Significant dates in the history of the Laws"
    sig_match = re.search(
        r"(Significant dates in the history of the Laws are as follows:)", raw
    )

    if sig_match:
        preface_text = raw[: sig_match.start()]
        history_text = raw[sig_match.end() :]
    else:
        preface_text = raw
        history_text = ""

    # Clean and add preface chunk
    preface_clean = clean_text(preface_text)
    # Remove trailing signature/address lines if present
    preface_clean = re.sub(
        r"\s*R\.S\. Lawson.*$", "", preface_clean, flags=re.DOTALL
    )
    if preface_clean:
        chunks.append({
            "id": "preface",
            "law_number": None,
            "law_title": "THE PREFACE",
            "section": None,
            "section_title": None,
            "text": preface_clean.strip(),
        })

    # Parse individual historical dates as separate chunks
    # Pattern: year followed by text (e.g., "1744   The earliest known...")
    date_pattern = re.compile(r"(\d{4})\s{2,}(.+?)(?=\d{4}\s{2,}|\Z)", re.DOTALL)
    for m in date_pattern.finditer(history_text):
        year = m.group(1)
        event_text = clean_text(m.group(2))
        if event_text:
            chunks.append({
                "id": f"history_{year}",
                "law_number": None,
                "law_title": "SIGNIFICANT DATES IN THE HISTORY OF THE LAWS",
                "section": year,
                "section_title": f"Year {year}",
                "text": f"{year} - {event_text}",
            })

    return chunks


def extract_content_text(reader):
    """Extract main content text (preamble, laws, appendices) with structural fixes."""
    raw = extract_pages(reader, CONTENT_PAGE_START, CONTENT_PAGE_END)
    # Fix missing newlines before LAW headings
    raw = re.sub(r"(?<!\n)(LAW\s+\d+\s+[A-Z])", r"\n\1", raw)
    # Fix missing newlines before APPENDIX headings
    raw = re.sub(r"(?<!\n)(APPENDIX\s+[A-E])", r"\n\1", raw)
    # Fix "GLOVESINDEX" concatenation
    raw = raw.replace("GLOVESINDEX", "GLOVES\nINDEX")
    # Remove INDEX section onwards
    index_pos = raw.find("\nINDEX\n")
    if index_pos != -1:
        raw = raw[:index_pos]
    raw = raw.rstrip() + "\n"
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw


def split_into_sections(raw_text):
    """Split raw content text into preamble, laws, and appendices."""

    law_pattern = re.compile(
        r"\n(LAW\s+(\d+)\s+([A-Z][A-Z\s\u2019';,\-/()]+?))\s*\n", re.MULTILINE
    )
    appendix_pattern = re.compile(
        r"\n(APPENDIX\s+([A-E])\s+(.+?))\s*(?:\n|$)", re.MULTILINE
    )

    boundaries = []

    for m in law_pattern.finditer(raw_text):
        law_num = int(m.group(2))
        if not any(b["number"] == law_num and b["type"] == "law" for b in boundaries):
            boundaries.append({
                "type": "law",
                "number": law_num,
                "title": m.group(3).strip(),
                "start": m.end(),
                "heading_start": m.start(),
            })

    for m in appendix_pattern.finditer(raw_text):
        boundaries.append({
            "type": "appendix",
            "number": m.group(2),
            "title": m.group(3).strip(),
            "start": m.end(),
            "heading_start": m.start(),
        })

    boundaries.sort(key=lambda b: b["start"])

    # Extract preamble
    preamble_match = re.search(
        r"THE PREAMBLE\s*[-\u2013]\s*THE SPIRIT OF CRICKET\s*\n", raw_text
    )
    sections = []
    if preamble_match and boundaries:
        preamble_text = raw_text[preamble_match.end() : boundaries[0]["heading_start"]]
        sections.append({
            "type": "preamble",
            "number": 0,
            "title": "THE PREAMBLE - THE SPIRIT OF CRICKET",
            "text": preamble_text.strip(),
        })

    for i, boundary in enumerate(boundaries):
        end = (
            boundaries[i + 1]["heading_start"]
            if i + 1 < len(boundaries)
            else len(raw_text)
        )
        text = raw_text[boundary["start"] : end].strip()
        sections.append({
            "type": boundary["type"],
            "number": boundary["number"],
            "title": boundary["title"],
            "text": text,
        })

    return sections


def chunk_section(section):
    """Split a section into deep sub-law chunks with parent context preserved."""
    chunks = []
    section_type = section["type"]
    number = section["number"]
    title = section["title"]
    text = section["text"]

    if section_type == "preamble":
        chunks.append({
            "id": "preamble",
            "law_number": 0,
            "law_title": title,
            "section": None,
            "section_title": None,
            "text": clean_text(text),
        })
        return chunks

    if section_type == "law":
        prefix = str(number)
        id_prefix = f"law{number}"
    else:
        prefix = str(number)
        id_prefix = f"appendix_{number.lower()}"

    # Split at depth N.N and N.N.N (e.g., 2.3 and 2.3.1) but NOT deeper.
    # N.N.N.N and below stay inside their parent N.N.N chunk for context.
    all_sublaw_pattern = re.compile(
        rf"^({re.escape(prefix)}\.\d+(?:\.\d+)?)\s{{2,}}(\S.+)", re.MULTILINE
    )
    all_splits = list(all_sublaw_pattern.finditer(text))

    if not all_splits:
        chunks.append({
            "id": id_prefix,
            "law_number": number,
            "law_title": title,
            "section": None,
            "section_title": None,
            "text": clean_text(text),
        })
        return chunks

    # Intro text before the first sub-law
    intro_text = clean_text(text[: all_splits[0].start()])
    if intro_text:
        chunks.append({
            "id": id_prefix,
            "law_number": number,
            "law_title": title,
            "section": None,
            "section_title": None,
            "text": intro_text,
        })

    for i, match in enumerate(all_splits):
        sub_num = match.group(1)
        sub_title = match.group(2).strip()
        start = match.start()
        end = all_splits[i + 1].start() if i + 1 < len(all_splits) else len(text)
        raw_chunk_text = text[start:end].strip()

        # Determine parent context: if this is N.N.N or deeper,
        # prepend the parent sub-law's intro paragraph for context
        depth = sub_num.count(".")
        if depth >= 2:
            # Find the parent (e.g., for 2.3.1, parent is 2.3)
            parent_num = sub_num.rsplit(".", 1)[0]
            parent_match = next(
                (s for s in all_splits if s.group(1) == parent_num), None
            )
            if parent_match:
                # Get the parent's intro text (text between parent heading and first child)
                parent_children = [
                    s
                    for s in all_splits
                    if s.group(1).startswith(parent_num + ".")
                    and s.group(1).count(".") == depth
                ]
                if parent_children:
                    parent_intro_end = parent_children[0].start()
                    parent_intro = clean_text(
                        text[parent_match.start() : parent_intro_end]
                    )
                    if parent_intro and not raw_chunk_text.startswith(parent_intro):
                        raw_chunk_text = parent_intro + " " + raw_chunk_text

        chunk_id = f"{id_prefix}_s{sub_num}"
        chunks.append({
            "id": chunk_id,
            "law_number": number,
            "law_title": title,
            "section": sub_num,
            "section_title": sub_title,
            "text": clean_text(raw_chunk_text),
        })

    return chunks


def main():
    download_pdf()
    reader = PdfReader(PDF_PATH)

    print("Extracting preface and history...")
    preface_chunks = extract_preface_and_history(reader)
    print(f"  {len(preface_chunks)} preface/history chunks")

    print("Extracting main content...")
    raw_text = extract_content_text(reader)
    print(f"  {len(raw_text)} characters of text")

    print("Splitting into laws and appendices...")
    sections = split_into_sections(raw_text)
    print(f"  {len(sections)} sections")

    print("Chunking sections into sub-laws...")
    content_chunks = []
    for section in sections:
        content_chunks.extend(chunk_section(section))

    # Filter out any empty chunks
    content_chunks = [c for c in content_chunks if c["text"]]
    all_chunks = preface_chunks + content_chunks
    law_count = len([s for s in sections if s["type"] == "law"])
    appendix_count = len([s for s in sections if s["type"] == "appendix"])

    output = {
        "metadata": {
            "source": "MCC Laws of Cricket (2017 Code 4th Edition - 2026)",
            "url": PDF_URL,
            "total_laws": law_count,
            "total_appendices": appendix_count,
            "total_chunks": len(all_chunks),
        },
        "chunks": all_chunks,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Saved {len(all_chunks)} chunks to {OUTPUT_PATH}")
    print(f"  Laws: {law_count}, Appendices: {appendix_count}")

    print("\nSample chunks:")
    for chunk in all_chunks[:5]:
        section_info = f" | {chunk['section']}" if chunk["section"] else ""
        print(f"  {chunk['id']}: {chunk['law_title']}{section_info}")
    print("  ...")
    for chunk in all_chunks[-3:]:
        section_info = f" | {chunk['section']}" if chunk["section"] else ""
        print(f"  {chunk['id']}: {chunk['law_title']}{section_info}")


if __name__ == "__main__":
    main()
