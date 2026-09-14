"""
utils/semantic_chunker.py
-------------------------
Semantic Chunker for ETD-Hub Knowledge Graph Pipeline.

This module splits long text (themes, questions, answers) into
semantically meaningful segments that improve LLM entity and
relation extraction accuracy.

Chunking Strategy:
    1. Header-based segmentation (Bias / Dataset / Governance / Risk)
    2. Semantic keyword-based segmentation
    3. Sentence-boundary chunking (250–450 token chunks)
    4. Fallback: token-limited chunks

Output:
    List[str] = list of text chunks
"""

import re
from typing import Any, List, Dict, Tuple
import tiktoken
from config import Config


class SemanticChunker:
    SAFE_LOCAL_ANAPHORA_PATTERN = re.compile(
        r"^(it|this system|these systems|these algorithms|such systems|this model|these tools|these predictions)\b",
        re.IGNORECASE,
    )
    SECTION_ALIASES = {
        "domain": "Domain",
        "description": "Description",
        "ethical challenges": "Ethical Challenges",
        "ethical challenge": "Ethical Challenges",
        "public datasets": "Public Datasets",
        "public dataset": "Public Datasets",
        "source": "Source",
        "url": "URL",
        "dashboard": "Dashboard",
        "github analysis": "GitHub Analysis",
        "research paper": "Research Paper",
        "content": "Content",
        "features": "Features",
        "labels": "Labels",
        "additional dataset": "Additional Dataset",
        "additional datasets": "Additional Dataset",
        "data portal": "Data Portal",
        "historical data": "Historical Data",
        "includes": "Includes",
        "primary": "Primary",
        "supplementary": "Supplementary",
        "question": "Question",
        "answer": "Answer",
        "body": "Body",
    }

    RESOURCE_SECTION_NAMES = {
        "Public Datasets",
        "Additional Dataset",
        "Source",
        "URL",
        "Dashboard",
        "GitHub Analysis",
        "Research Paper",
        "Content",
        "Features",
        "Labels",
        "Data Portal",
        "Historical Data",
        "Primary",
        "Supplementary",
        "Includes",
    }

    SEMANTIC_SECTION_NAMES = {
        "Description",
        "Ethical Challenges",
        "Question",
        "Answer",
        "Body",
    }

    # ============================================================
    # 1. Canonical topic headers (very common in your JSON dataset)
    # ============================================================
    TOPIC_HEADERS = [
        r"ethical challenge[s]?",
        r"fairness",
        r"bias",
        r"dataset",
        r"data[- ]set",
        r"risk",
        r"governance",
        r"privacy",
        r"transparency",
        r"explainability",
        r"accountability",
        r"context",
        r"methodology",
        r"problem",
        r"motivation",
        r"conclusion",
        r"summary",
    ]

    # ============================================================
    # 2. Keyword-based soft boundaries
    # ============================================================
    SEMANTIC_CUES = [
        "in this dataset",
        "the main bias",
        "the key risk",
        "the primary risk",
        "ethical concern",
        "fairness concern",
        "according to",
        "research shows",
        "for example",
        "case study",
        "in practice",
        "in reality",
        "evaluation",
        "metric",
        "governance",
        "policy",
        "regulation",
        "human oversight",
        "decision-making",
    ]

    # ============================================================
    # Tokenizer for GPT-4o-mini
    # ============================================================
    try:
        ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        ENCODER = None

    @staticmethod
    def count_tokens(text: str) -> int:
        """Approximate token count for GPT models."""
        if SemanticChunker.ENCODER:
            return len(SemanticChunker.ENCODER.encode(text))
        # fallback approximation
        return max(1, int(len(text.split()) * 1.3))

    # ============================================================
    # MAIN ENTRY: chunk()
    # ============================================================
    @classmethod
    def chunk(cls, text: str) -> List[str]:
        """
        Master chunking function.
        Returns a list of semantically meaningful segments.
        """
        return [x["text"] for x in cls.chunk_with_offsets(text)]

    @staticmethod
    def _trim_span(text: str, start: int, end: int) -> Tuple[int, int]:
        s = int(max(0, start))
        e = int(max(s, end))
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        return s, e

    @classmethod
    def _chunk_by_topic_headers_with_offsets(cls, text: str) -> List[Tuple[int, int]]:
        header_patterns = [fr"(?:^|\n)\s*({header})\s*:?" for header in cls.TOPIC_HEADERS]
        pattern = r"(?im)" + "|".join(header_patterns)
        matches = list(re.finditer(pattern, text))
        if not matches:
            return [(0, len(text))]

        spans: List[Tuple[int, int]] = []
        prev = 0
        for m in matches:
            start = m.start()
            if start > 0 and text[start] == "\n":
                start += 1
            if start - prev > 30:
                spans.append((prev, start))
            prev = start
        spans.append((prev, len(text)))
        return spans

    @classmethod
    def _split_segment_by_tokens_with_offsets(cls, text: str, seg_start: int, seg_end: int) -> List[Tuple[int, int]]:
        seg_text = text[seg_start:seg_end]
        if not seg_text.strip():
            return []

        MAX_TOK = 450
        if cls.count_tokens(seg_text.strip()) <= MAX_TOK:
            s, e = cls._trim_span(text, seg_start, seg_end)
            return [(s, e)] if e > s else []

        sentence_pattern = re.compile(r"[^.!?]+(?:[.!?]+|$)\s*", flags=re.S)
        sentence_spans: List[Tuple[int, int]] = []
        for m in sentence_pattern.finditer(seg_text):
            s = seg_start + m.start()
            e = seg_start + m.end()
            ts, te = cls._trim_span(text, s, e)
            if te > ts:
                sentence_spans.append((ts, te))

        if not sentence_spans:
            ts, te = cls._trim_span(text, seg_start, seg_end)
            return [(ts, te)] if te > ts else []

        out: List[Tuple[int, int]] = []
        cur_start: int = sentence_spans[0][0]
        cur_end: int = sentence_spans[0][1]

        for s, e in sentence_spans[1:]:
            candidate = text[cur_start:e]
            if cls.count_tokens(candidate) <= MAX_TOK:
                cur_end = e
            else:
                out.append((cur_start, cur_end))
                if cls.count_tokens(text[s:e]) > MAX_TOK:
                    words = list(re.finditer(r"\S+\s*", text[s:e]))
                    if not words:
                        out.append((s, e))
                        cur_start, cur_end = s, e
                        continue
                    approx_words = max(1, int(MAX_TOK / 1.3))
                    i = 0
                    while i < len(words):
                        ws = s + words[i].start()
                        j = min(len(words), i + approx_words) - 1
                        we = s + words[j].end()
                        ts, te = cls._trim_span(text, ws, we)
                        if te > ts:
                            out.append((ts, te))
                        i = j + 1
                    cur_start, cur_end = s, e
                else:
                    cur_start, cur_end = s, e
        out.append((cur_start, cur_end))
        return out

    @classmethod
    def _merge_small_chunks_with_offsets(
        cls,
        text: str,
        spans: List[Tuple[int, int]],
        min_tok: int = 200,
        max_tok: int = 450,
    ) -> List[Tuple[int, int]]:
        merged: List[Tuple[int, int]] = []
        if not spans:
            return merged
        cur_s, cur_e = spans[0]
        for s, e in spans[1:]:
            candidate_s = cur_s
            candidate_e = e
            cur_text = text[cur_s:cur_e]
            candidate_text = text[candidate_s:candidate_e]
            if cls.count_tokens(candidate_text) <= max_tok and cls.count_tokens(cur_text) < min_tok:
                cur_e = candidate_e
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        return merged

    @classmethod
    def chunk_with_offsets(cls, text: str) -> List[Dict[str, Any]]:
        if text is None:
            return []
        source_text = str(text)
        if not source_text.strip():
            return []

        if len(source_text.strip()) < 80:
            s, e = cls._trim_span(source_text, 0, len(source_text))
            return [{"text": source_text[s:e], "start_offset": int(s), "end_offset": int(e), "chunk_index": 0}] if e > s else []

        header_spans = cls._chunk_by_topic_headers_with_offsets(source_text)
        token_spans: List[Tuple[int, int]] = []
        for hs, he in header_spans:
            token_spans.extend(cls._split_segment_by_tokens_with_offsets(source_text, hs, he))
        token_spans = cls._merge_small_chunks_with_offsets(source_text, token_spans, min_tok=200, max_tok=450)

        out: List[Dict[str, int]] = []
        for idx, (s, e) in enumerate(token_spans):
            ts, te = cls._trim_span(source_text, s, e)
            if te <= ts:
                continue
            out.append(
                {
                    "text": source_text[ts:te],
                    "start_offset": int(ts),
                    "end_offset": int(te),
                    "chunk_index": int(idx),
                }
            )
        return out

    @classmethod
    def _normalize_section_name(cls, raw_name: str) -> str:
        key = re.sub(r"\s+", " ", str(raw_name or "").strip()).casefold()
        return cls.SECTION_ALIASES.get(key, str(raw_name or "").strip() or "Body")

    @classmethod
    def _section_category(cls, name: str, field: str = "") -> str:
        canon = cls._normalize_section_name(name)
        include_resource = bool(getattr(Config, "INCLUDE_RESOURCE_SECTIONS_IN_SEMANTICS", True))
        if canon in cls.RESOURCE_SECTION_NAMES and not include_resource:
            return "resource"
        if canon in cls.SEMANTIC_SECTION_NAMES:
            return "semantic"
        field_norm = str(field or "").strip()
        if field_norm in {"question.body", "answer.description"}:
            return "semantic"
        if canon == "Domain":
            return "metadata"
        return "semantic"

    @classmethod
    def parse_structured_sections(cls, text: str, field: str = "") -> List[Dict[str, Any]]:
        if text is None:
            return []
        source_text = str(text)
        if not source_text:
            return []

        heading_pattern = re.compile(r"(?m)^\s*\*\*(?P<name>[^*\n:]+?)\*\*\s*:\s*(?P<rest>.*)$")
        matches = list(heading_pattern.finditer(source_text))
        sections: List[Dict[str, Any]] = []

        if not matches:
            name = "Body" if str(field or "").strip() != "theme.description" else "Description"
            return [
                {
                    "section": name,
                    "section_category": cls._section_category(name, field),
                    "start_offset": 0,
                    "end_offset": len(source_text),
                    "text": source_text,
                }
            ]

        for idx, match in enumerate(matches):
            raw_name = match.group("name")
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(source_text)
            canon = cls._normalize_section_name(raw_name)
            sections.append(
                {
                    "section": canon,
                    "section_category": cls._section_category(canon, field),
                    "start_offset": int(start),
                    "end_offset": int(end),
                    "text": source_text[start:end],
                    "section_text": source_text[start:end],
                }
            )
        return sections

    @classmethod
    def _split_section_into_local_units(cls, section: Dict[str, Any], field: str = "") -> List[Dict[str, Any]]:
        text = str(section.get("text") or "")
        if not text.strip():
            return []

        base_start = int(section.get("start_offset", 0) or 0)
        section_name = str(section.get("section") or "Body")
        section_category = str(section.get("section_category") or cls._section_category(section_name, field))
        units: List[Dict[str, Any]] = []
        heading_match = re.match(r"^\s*\*\*[^*\n:]+?\*\*\s*:\s*(?P<body>.*)$", text, flags=re.S)
        if heading_match:
            body = str(heading_match.group("body") or "")
            if body.strip():
                base_start += int(heading_match.start("body"))
                text = body
        single_line = "\n" not in text.strip()
        looks_like_url_or_resource = bool(re.search(r"https?://", text)) or bool(re.fullmatch(r"[^.!?]+", text.strip()))
        has_multiple_sentences = bool(re.search(r"[.!?]\s+[A-Z]", text.strip()))
        if single_line and looks_like_url_or_resource and not has_multiple_sentences:
            trimmed_start, trimmed_end = cls._trim_span(text, 0, len(text))
            if trimmed_end > trimmed_start:
                return [
                    {
                        "text": text[trimmed_start:trimmed_end],
                        "start_offset": base_start + int(trimmed_start),
                        "end_offset": base_start + int(trimmed_end),
                        "section": section_name,
                        "section_category": section_category,
                        "section_text": text,
                        "unit_type": "section",
                        "unit_id": f"{section_name}:0",
                    }
                ]

        bullet_pattern = re.compile(r"(?m)^(?P<indent>\s*(?:[-*]|\d+[.)])\s+)(?P<body>.+?)(?=(?:\n\s*(?:[-*]|\d+[.)])\s+)|\Z)", re.S)
        bullet_matches = list(bullet_pattern.finditer(text))
        if bullet_matches:
            for idx, match in enumerate(bullet_matches):
                raw_unit = match.group("body")
                local_start = match.start("body")
                local_end = match.end("body")
                abs_start, abs_end = cls._trim_span(text, local_start, local_end)
                if abs_end <= abs_start:
                    continue
                unit_text = text[abs_start:abs_end]
                units.append(
                    {
                        "text": unit_text,
                        "start_offset": base_start + int(abs_start),
                        "end_offset": base_start + int(abs_end),
                        "section": section_name,
                        "section_category": section_category,
                        "section_text": text,
                        "unit_type": "bullet",
                        "unit_id": f"{section_name}:{idx}",
                    }
                )
            return units

        sentence_pattern = re.compile(r"[^.!?\n]+(?:[.!?]+|$)", re.S)
        sent_idx = 0
        for match in sentence_pattern.finditer(text):
            local_start = match.start()
            local_end = match.end()
            abs_start, abs_end = cls._trim_span(text, local_start, local_end)
            if abs_end <= abs_start:
                continue
            unit_text = text[abs_start:abs_end]
            if not unit_text.strip():
                continue
            units.append(
                {
                    "text": unit_text,
                    "start_offset": base_start + int(abs_start),
                    "end_offset": base_start + int(abs_end),
                    "section": section_name,
                    "section_category": section_category,
                    "section_text": text,
                    "unit_type": "sentence",
                    "unit_id": f"{section_name}:{sent_idx}",
                }
            )
            sent_idx += 1

        if units:
            units = cls._merge_abbreviation_units(units)
            for idx, unit in enumerate(units):
                unit["unit_id"] = f"{section_name}:{idx}"
            return cls._annotate_safe_local_anaphora(units)

        trimmed_start, trimmed_end = cls._trim_span(text, 0, len(text))
        if trimmed_end <= trimmed_start:
            return []
        return [
            {
                "text": text[trimmed_start:trimmed_end],
                "start_offset": base_start + int(trimmed_start),
                "end_offset": base_start + int(trimmed_end),
                "section": section_name,
                "section_category": section_category,
                "section_text": text,
                "unit_type": "section",
                "unit_id": f"{section_name}:0",
            }
        ]

    @classmethod
    def _annotate_safe_local_anaphora(cls, units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not units:
            return units
        for idx, unit in enumerate(units):
            unit_text = str(unit.get("text") or "").strip()
            if idx == 0 or not unit_text:
                continue
            match = cls.SAFE_LOCAL_ANAPHORA_PATTERN.match(unit_text)
            if not match:
                continue
            prev = units[idx - 1]
            prev_text = str(prev.get("text") or "").strip()
            if not prev_text:
                continue
            unit["anaphora"] = {
                "source_unit_id": str(prev.get("unit_id") or ""),
                "source_start_offset": int(prev.get("start_offset") or 0),
                "source_end_offset": int(prev.get("end_offset") or 0),
                "surface_form": match.group(0),
                "source": "preceding_sentence",
            }
        return units

    @staticmethod
    def _merge_abbreviation_units(units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not units:
            return units
        merged: List[Dict[str, Any]] = []
        buffer = dict(units[0])
        for unit in units[1:]:
            prev_text = str(buffer.get("text") or "").strip()
            curr_text = str(unit.get("text") or "").strip()
            if re.fullmatch(r"[A-Z]\.", curr_text) or re.search(r"\b[A-Z]\.$", prev_text):
                buffer["text"] = f"{str(buffer.get('text') or '')}{str(unit.get('text') or '')}"
                buffer["end_offset"] = int(unit.get("end_offset") or buffer.get("end_offset") or 0)
                continue
            merged.append(buffer)
            buffer = dict(unit)
        merged.append(buffer)
        return merged

    @classmethod
    def semantic_relation_units(cls, text: str, field: str = "") -> List[Dict[str, Any]]:
        sections = cls.parse_structured_sections(text, field=field)
        units: List[Dict[str, Any]] = []
        for section in sections:
            if str(section.get("section_category") or "") != "semantic":
                continue
            units.extend(cls._split_section_into_local_units(section, field=field))
        return units


    # ============================================================
    # STEP 1 — HEADER-BASED CHUNKING
    # ============================================================
    @classmethod
    def _chunk_by_topic_headers(cls, text: str) -> List[str]:
        """
        Splits text on logical topic sections such as:
        "Ethical Challenges", "Dataset", "Fairness", "Risk".
        """

        # Move (?i) to the front of the full expression, not each fragment
        # Treat headers as headings only when they appear at line starts or after a newline,
        # optionally followed by ":" (reduces splitting inside normal sentences).
        header_patterns = [fr"(?:^|\n)\s*({header})\s*:?" for header in cls.TOPIC_HEADERS]
        pattern = r"(?im)" + "|".join(header_patterns)

        matches = list(re.finditer(pattern, text))
        if not matches:
            return [text]

        chunks = []
        prev = 0

        for m in matches:
            start = m.start()
            # keep newline boundary clean
            if start > 0 and text[start] == "\n":
                start += 1            
            
            if start - prev > 30:  # avoid tiny gaps
                chunks.append(text[prev:start])
            prev = start

        chunks.append(text[prev:])
        
        return chunks



    # ============================================================
    # STEP 2 — SOFT SEMANTIC CUE SPLITTING
    # ============================================================
    @classmethod
    def _chunk_by_semantic_cues(cls, segments: List[str]) -> List[str]:
        """
         Further splits segments based on soft semantic cues
        (e.g., 'ethical concern', 'in this dataset', 'evaluation').
        Uses case-insensitive splitting WITHOUT inline regex flags.
        """
        new_segments = []

        # Build alternation pattern WITHOUT (?i)
        cue_pattern = "|".join([re.escape(cue) for cue in cls.SEMANTIC_CUES])

        for segment in segments:
            # Apply case-insensitive splitting using flags
            splits = re.split(cue_pattern, segment, flags=re.IGNORECASE)
            if len(splits) > 1:
                new_segments.extend(splits)
            else:
                new_segments.append(segment)

        return new_segments


    # ============================================================
    # STEP 3 — TOKEN-SAFE CHUNKING
    # ============================================================
    @classmethod
    def _split_by_tokens(cls, segments: List[str]) -> List[str]:
        """
        Ensures chunks are between 200–450 tokens.
        Preserves sentence boundaries.
        """
        MIN_TOK = 200
        MAX_TOK = 450

        final_chunks = []

        for seg in segments:
            seg = seg.strip()
            tok = cls.count_tokens(seg)

            if tok <= MAX_TOK:
                final_chunks.append(seg)
                continue

            # Split into sentences
            sentences = re.split(r"(?<=[.!?])\s+", seg)
            buffer = ""

            for sent in sentences:
                if not sent.strip():
                    continue

                new_buffer = buffer + " " + sent if buffer else sent
                if cls.count_tokens(new_buffer) > MAX_TOK:
                    # flush and reset
                    if buffer:
                        final_chunks.append(buffer)
                        buffer = sent
                    else:
                        # extremely long sentence; hard-split by tokens
                        if cls.ENCODER:
                            toks = cls.ENCODER.encode(sent)
                            for i in range(0, len(toks), MAX_TOK):
                                piece = cls.ENCODER.decode(toks[i:i + MAX_TOK]).strip()
                                if piece:
                                    final_chunks.append(piece)
                        else:
                            # fallback: split by words
                            words = sent.split()
                            approx_words = max(1, int(MAX_TOK / 1.3))
                            for i in range(0, len(words), approx_words):
                                piece = " ".join(words[i:i + approx_words]).strip()
                                if piece:
                                    final_chunks.append(piece)
                        buffer = ""
                else:
                    buffer = new_buffer

            if buffer:
                final_chunks.append(buffer)

        return final_chunks


    @classmethod
    def _merge_small_chunks(cls, chunks: List[str], min_tok: int = 200, max_tok: int = 450) -> List[str]:
        """Merge adjacent small chunks to reach min_tok without exceeding max_tok."""
        merged: List[str] = []
        buf = ""

        for ch in chunks:
            ch = ch.strip()
            if not ch:
                continue

            if not buf:
                buf = ch
                continue

            # try merge
            candidate = buf + "\n" + ch
            if cls.count_tokens(candidate) <= max_tok and cls.count_tokens(buf) < min_tok:
                buf = candidate
            else:
                merged.append(buf)
                buf = ch

        if buf:
            merged.append(buf)

        return merged
