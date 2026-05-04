# app/components/document_processing/services/classifier_service.py

import json
import logging
import difflib
import re
from app.core.config import settings
from app.services.evaluation.gemini_cost_policy import EvaluationGeminiClient
from app.shared.ai.gemini_client import gemini_generate, gemini_generate_evaluation, gemini_generate_lightweight

logger = logging.getLogger(__name__)

# Define prompts and constants
CLASSIFY_PROMPT = """
You are a Sinhala educational document classifier.
Classify the following text into ONE category:
- term_test
- teacher_guide
- student_notes
- past_paper
- answer_scheme
- textbook
Respond with ONLY the category name.
TEXT:
{content}
"""

def classify_document(text: str) -> str:
    if not text or not text.strip():
        return "unknown"
    
    try:
        output = gemini_generate(CLASSIFY_PROMPT.format(content=text[:8000])).strip().lower()
        
        allowed = {
            "term_test", "teacher_guide", "student_notes",
            "past_paper", "answer_scheme", "textbook"
        }
        # Print for debugging
        print(f"Document classified as: {output}")
        return output if output in allowed else "unknown"

    except Exception as e:
        print(f"Error classifying document: {e}")
        return "unknown"

def _safe_json_loads(text: str):
    """Robust JSON parsing that handles Markdown code blocks and truncated JSON."""
    if not text:
        return {}
    
    try:
        # Strip potential Markdown backticks
        clean_text = text.strip()
        if clean_text.startswith("```"):
            clean_text = re.sub(r'^```json\s*|\s*```$', '', clean_text, flags=re.MULTILINE)
        
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError:
            # Try to repair truncated JSON
            repaired = _repair_json(clean_text)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                recovered = _recover_largest_valid_json(clean_text)
                if recovered is not None:
                    return recovered
                raise
    except Exception as e:
        logger.error(f"Failed to parse JSON response: {e}. Raw content: {text[:500]}...")
        return {}


def _recover_largest_valid_json(text: str):
    """
    Recover the largest complete JSON object/list inside a malformed response.
    This handles Gemini truncation or trailing prose without throwing away all
    mappings that were already emitted.
    """
    if not text:
        return None

    decoder = json.JSONDecoder()
    best = None
    best_span = -1
    starts = [idx for idx, ch in enumerate(text) if ch in "[{"]

    for start in starts:
        snippet = text[start:].strip()
        try:
            value, end = decoder.raw_decode(snippet)
            if end > best_span:
                best = value
                best_span = end
        except json.JSONDecodeError:
            pass

        repaired = _repair_json(snippet)
        try:
            value = json.loads(repaired)
            span = len(repaired)
            if span > best_span:
                best = value
                best_span = span
        except Exception:
            continue

    return best

def _repair_json(text: str) -> str:
    """Basic recovery for truncated JSON (unterminated strings/objects)."""
    text = text.strip()
    if not text: return "{}"

    # Remove trailing commas before closing braces/brackets, which Gemini
    # occasionally emits even in otherwise-valid JSON payloads.
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    text = re.sub(r",\s*$", "", text)
    
    # 1. Close unterminated string if it ends abruptly
    # If the last character is NOT a quote but there's an odd number of quotes in the last line
    # or if it ends with backslash escape
    if text.count('"') % 2 != 0:
        text += '"'
    
    # 2. Close open braces/brackets
    open_braces = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')
    
    # Add closing markers in reverse order of opening
    # This is a naive stack-based approach
    stack = []
    for char in text:
        if char == '{': stack.append('}')
        elif char == '[': stack.append(']')
        elif char == '}': 
            if stack and stack[-1] == '}': stack.pop()
        elif char == ']':
            if stack and stack[-1] == ']': stack.pop()
    
    # Append missing closers
    while stack:
        text += stack.pop()

    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return text


_ROMAN_NUMERAL_MAP = {
    "i": 1,
    "ii": 2,
    "iii": 3,
    "iv": 4,
    "v": 5,
}


def _roman_to_paper_key(token: str) -> str | None:
    normalized = str(token or "").strip().lower()
    normalized = {"1": "i", "2": "ii", "3": "iii", "4": "iv", "5": "v"}.get(normalized, normalized)
    value = _ROMAN_NUMERAL_MAP.get(normalized)
    if value is None:
        return None
    return f"Paper_{normalized.upper()}"


def _question_sort_key(label: str) -> tuple[int, str]:
    text = str(label or "").strip()
    match = re.search(r"\d+", text)
    return (int(match.group()) if match else 10**9, text)


def _question_has_sub_questions(question_data: dict) -> bool:
    return bool((question_data or {}).get("sub_questions"))


def _count_questions_with_subparts(questions: dict) -> int:
    return sum(1 for q in (questions or {}).values() if _question_has_sub_questions(q))


_PAPER_HEADER_WORDS = (
    "paper",
    "part",
    "section",
    "කොටස",
    "කොටස",
    "කාටස",
    "ශ්‍රකාටස",
    "ශ්‍රේකාටස",
    "භාගය",
)
_PAPER_HEADER_WORD_PATTERN = "|".join(re.escape(word) for word in _PAPER_HEADER_WORDS)
_PAPER_TOKEN_PATTERN = r"iii|ii|iv|v|i|[1-5]"


def _detect_paper_header_key(line: str) -> str | None:
    if not line or len(line.strip()) > 160:
        return None

    normalized = re.sub(r"\s+", " ", line.strip().lower())
    patterns = [
        rf"^(?P<token>{_PAPER_TOKEN_PATTERN})\s*[-:.)]?\s*(?:{_PAPER_HEADER_WORD_PATTERN})\b",
        rf"^(?:{_PAPER_HEADER_WORD_PATTERN})\s*[-: ]*\s*(?P<token>{_PAPER_TOKEN_PATTERN})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            return _roman_to_paper_key(match.group("token"))
    return None


def _segment_exam_text_by_paper_headers(text: str) -> dict[str, str]:
    if not text:
        return {}

    matches: list[tuple[str, int]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        paper_key = _detect_paper_header_key(line)
        if paper_key:
            matches.append((paper_key, offset))
        offset += len(line)

    if len(matches) >= 2:
        sections: dict[str, str] = {}
        for index, (paper_key, start) in enumerate(matches):
            end = matches[index + 1][1] if index + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            if section_text:
                sections[paper_key] = section_text
        return sections

    header_pattern = re.compile(
        r"(?im)^\s*(?:"
        r"(?:(?:paper|part|section)\s*[-: ]*\s*(?P<leading>iii|ii|iv|v|i))"
        r"|(?:(?P<trailing>iii|ii|iv|v|i)\s*(?:paper|part|section|කොටස|භාගය))"
        r"|(?:(?:කොටස|භාගය)\s*(?P<sinhala>iii|ii|iv|v|i))"
        r")\b[^\n]*$"
    )

    matches = list(header_pattern.finditer(text))
    if len(matches) < 2:
        return {}

    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        token = match.group("leading") or match.group("trailing") or match.group("sinhala")
        paper_key = _roman_to_paper_key(token)
        if not paper_key:
            continue
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections[paper_key] = section_text
    return sections


def _prepare_exam_text_for_prompt(text: str) -> str:
    sections = _segment_exam_text_by_paper_headers(text)
    if len(sections) < 2:
        return text

    ordered = sorted(sections.items(), key=lambda item: _question_sort_key(item[0]))
    blocks = [
        f"{paper_key} SECTION (do not mix with other papers):\n{section_text}"
        for paper_key, section_text in ordered
    ]
    return (
        "The OCR text below was locally segmented by detected paper headers. "
        "Keep every question inside its own paper section and do not move questions across sections. "
        f"You MUST return these paper keys: {', '.join(key for key, _ in ordered)}.\n\n"
        + "\n\n".join(blocks)
    )


def _extract_structured_main_questions_from_section(section_text: str) -> dict[str, dict]:
    if not section_text:
        return {}

    marker_pattern = re.compile(r"(?m)^\s*(?P<label>\d{1,2})\.\s*$")
    matches = list(marker_pattern.finditer(section_text))
    questions: dict[str, dict] = {}

    for index, match in enumerate(matches):
        raw_label = match.group("label")
        label = str(int(raw_label))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(section_text)
        body = section_text[start:end].strip()
        if not body:
            continue

        sub_questions = {}
        for sub_match in re.finditer(
            r"(?ms)\(\s*(?P<label>[^)\s]{1,3})\s*\)\s*(?P<text>.*?)(?=\n\s*\(\s*[^)\s]{1,3}\s*\)|\Z)",
            body,
        ):
            sub_label = sub_match.group("label").strip()
            sub_text = re.sub(r"\s+", " ", sub_match.group("text")).strip()
            marks_match = re.search(r"\((\d{1,2})\s*[^\)]*ලකුණු\)", sub_text)
            sub_questions[sub_label] = {
                "text": sub_text,
                "marks": int(marks_match.group(1)) if marks_match else None,
            }

        questions[label] = {
            "type": "structured",
            "text": re.sub(r"\s+", " ", body).strip(),
            "marks": sum(
                value["marks"] for value in sub_questions.values() if value.get("marks") is not None
            ) or None,
            "sub_questions": sub_questions,
        }

    return questions


def _backfill_missing_segmented_questions(result: dict, original_text: str) -> dict:
    sections = _segment_exam_text_by_paper_headers(original_text)
    if not sections:
        return result

    for paper_key, section_text in sections.items():
        local_questions = _extract_structured_main_questions_from_section(section_text)
        if not local_questions:
            continue

        paper = result.setdefault(paper_key, {})
        if not isinstance(paper, dict):
            continue
        questions = paper.setdefault("questions", {})
        if not isinstance(questions, dict):
            questions = {}
            paper["questions"] = questions

        missing = {
            label: question
            for label, question in local_questions.items()
            if str(label) not in {str(existing) for existing in questions}
        }
        if missing:
            questions.update(missing)
            logger.warning(
                "Backfilled %s missing main question(s) for %s from locally segmented text.",
                len(missing),
                paper_key,
            )

        config = paper.setdefault("config", {})
        if isinstance(config, dict):
            config["total_questions_available"] = len(questions)

    return result


def _normalize_extracted_exam_result(result: dict) -> dict:
    cleaned_result: dict[str, dict] = {}
    for paper_key, paper_data in (result or {}).items():
        if not str(paper_key).startswith("Paper_") or not isinstance(paper_data, dict):
            continue

        questions = paper_data.get("questions", {}) or {}
        if not isinstance(questions, dict):
            questions = {}
        ordered_questions = {
            str(label): value
            for label, value in sorted(questions.items(), key=lambda item: _question_sort_key(item[0]))
        }

        config = dict(paper_data.get("config", {}) or {})
        config["total_questions_available"] = len(ordered_questions)

        normalized_paper = dict(paper_data)
        normalized_paper["config"] = config
        normalized_paper["questions"] = ordered_questions
        cleaned_result[str(paper_key)] = normalized_paper

    paper_i = cleaned_result.get("Paper_I")
    paper_ii = cleaned_result.get("Paper_II")
    if paper_i and paper_ii:
        paper_i_questions = dict(paper_i.get("questions", {}) or {})
        paper_ii_questions = dict(paper_ii.get("questions", {}) or {})
        paper_i_subparts = _count_questions_with_subparts(paper_i_questions)
        paper_ii_subparts = _count_questions_with_subparts(paper_ii_questions)

        # Heuristic repair for the common Sri Lankan split:
        # Paper I is short-answer/no subparts, Paper II is main questions with subparts.
        if paper_i_subparts == 0 and paper_ii_subparts >= 3:
            migrated: dict[str, dict] = {}
            remaining_paper_ii: dict[str, dict] = {}
            for label, question_data in paper_ii_questions.items():
                if _question_has_sub_questions(question_data):
                    remaining_paper_ii[label] = question_data
                else:
                    migrated[label] = question_data

            if migrated and remaining_paper_ii:
                paper_i_questions.update(migrated)
                paper_i["questions"] = {
                    label: value
                    for label, value in sorted(paper_i_questions.items(), key=lambda item: _question_sort_key(item[0]))
                }
                paper_ii["questions"] = {
                    label: value
                    for label, value in sorted(remaining_paper_ii.items(), key=lambda item: _question_sort_key(item[0]))
                }
                paper_i["config"]["total_questions_available"] = len(paper_i["questions"])
                paper_ii["config"]["total_questions_available"] = len(paper_ii["questions"])
                logger.warning(
                    "Rebalanced extracted paper structure: moved %s plain questions from Paper_II to Paper_I.",
                    len(migrated),
                )

    return cleaned_result
    

# 1. Specialized Prompt for Sinhala Exam Extraction
SINHALA_STRUCTURE_PROMPT = """
You are an expert AI for analyzing Sri Lankan exam papers (Sinhala and English medium).
Your task is to structure the provided text into a strict JSON format containing ONLY the question structure.

========================
IMPORTANT MARKING RULES
========================
- In Sri Lankan exam papers, NO question or subquestion has 0 marks.
- If marks are NOT clearly specified, use null (not 0).
- NEVER assign marks = 0.
- Common mark indicators: "ලකුණු", "marks", numbers in parentheses like (05)

========================
PAPER & NUMBERING RULES
========================
- Sri Lankan exam papers are divided into parts such as:
  - Part I (usually MCQs) - "I කොටස", "Part I", "භාගය I"
  - Part II (Structured questions) - "II කොටස", "Part II", "භාගය II"
  - Part III (Essay questions) - "III කොටස", "Part III", "භාගය III"

- EACH paper has its OWN numbering system.
  - Paper I numbering: 1, 2, 3, ... (typically 1-40 for MCQs)
  - Paper II numbering: starts again from 1 or continues with own sequence
  - DO NOT continue numbering across papers

- Questions MUST be grouped under their correct paper.
- NEVER mix Paper I and Paper II questions in the same numbering scope.

========================
MCQ STRUCTURE RULES (Paper I)
========================
- MCQs usually belong to Paper I unless explicitly stated otherwise.
- Detect MCQs by options like:
  - (1)(2)(3)(4), (A)(B)(C)(D), (අ)(ආ)(ඉ)(ඊ)
  - Numbered list format: 1. ... 2. ... 3. ... 4. ...
- If multiple MCQs share common instructions or data:
  - Attach the shared information to the first question using "shared_stem"
  - Subsequent questions reference it using "inherits_shared_stem_from"
- Each MCQ MUST include:
  - "type": "mcq"
  - "text": question text
  - "options": array of option texts
  - "marks": usually null or 1

========================
STRUCTURED QUESTIONS (Paper II)
========================
- Main questions: numbered (1, 2, 3, etc.) or labeled (Q01, Q02, etc.)
- Sub-questions: labeled with letters (a, b, c) or (i, ii, iii) or (අ, ආ, ඉ)
- Extract marks from patterns:
  - "(05 ලකුණු)", "(5 marks)", "(10)", "05 ලකුණු"
- If a main question has sub-questions:
  - Main question "marks" = sum of sub-question marks (or null if unclear)
  - Each sub-question should have its own marks

========================
TEXT PATTERNS TO DETECT
========================
**Question Numbers:**
- "01.", "1)", "ප්‍රශ්නය 01", "Question 1"
- Look for consistent numbering patterns

**Sub-question Labels:**
- "(අ)", "(ආ)", "(ඉ)", "(a)", "(b)", "(c)", "(i)", "(ii)"

**Marks Patterns:**
- "(05 ලකුණු)", "(5 marks)", "05 ලකුණු", "5 marks"
- Extract the number before "ලකුණු" or "marks"

**Section Headers:**
- "I කොටස", "II කොටස", "Part A", "Part B"
- "බහුවරණ ප්‍රශ්න" (MCQs), "ව්‍යුහගත ප්‍රශ්න" (Structured)

========================
OUTPUT FORMAT (STRICT JSON)
========================

{{
  "PaperStructure": {{
    "Paper_I": {{
      "type": "MCQ",
      "questions": {{
        "1": {{
          "type": "mcq",
          "text": "question text",
          "options": ["option 1", "option 2", "option 3", "option 4"],
          "correct_answer": "2",
          "marks": 1
        }}
      }}
    }},
    "Paper_II": {{
      "type": "Structured",
      "questions": {{ ... }}
    }},
    "Paper_III": {{
      "type": "Structured",
      "questions": {{ ... }}
    }}
  }}
}}

========================
IMPORTANT NOTES
========================
1. Do NOT extract metadata (Subject, Year, Grade).
2. Do NOT extract general instructions.
3. Only extract the Question Structure.
4. If a paper section is not found in the text, return an empty questions object for that paper.
5. Preserve original question numbers/labels from the text.
6. Extract marks carefully - use null if uncertain, never use 0.

========================
TEXT TO PROCESS
========================
{content}
"""

def separate_paper_content(text: str):
    """
    Separates paper content into PaperStructure only.
    Returns: paper_structure (dict)
    """
    if not text or not text.strip():
        return {}

    try:
        logger.info("Starting Sinhala structure extraction.")
        response_text = gemini_generate(
            SINHALA_STRUCTURE_PROMPT.format(content=text[:20000]),
            json_mode=True
        )

        if not response_text:
            logger.error("Empty response from Gemini.")
            return {}

        result = _safe_json_loads(response_text)
        if not result:
            return {}
        
        logger.info("Sinhala structure extraction completed successfully.")

        paper_structure = result.get("PaperStructure", {})
        if not paper_structure:
            # Fallback for older model styles or flat results
            paper_structure = {k: v for k, v in result.items() if k.startswith("Paper_")}

        # 🔒 Defensive normalization
        # Ensure 'questions' key exists for all identified papers
        for paper_key, paper in paper_structure.items():
            if isinstance(paper, dict) and "questions" not in paper:
                paper["questions"] = {}

        logger.debug("Extracted Paper Structure: %s", paper_structure)
        
        return paper_structure

    except json.JSONDecodeError:
        print("❌ Error: Model output was not valid JSON.")
        return {}

    except Exception as e:
        print(f"❌ Error in Sinhala structure extraction: {e}")
        return {}
    
def fix_sinhala_ocr(text: str) -> str:
    """
    Fix OCR errors in Sinhala text such as broken ligatures, misplaced diacritics,
    or split grapheme clusters.
    """
    if not text or not text.strip():
        return text

    prompt = f"""
    You are a Sinhala OCR text corrector.
    Fix OCR errors such as:
    - broken conjunct letters (ex: ක් ෂ → ක්‍ෂ),
    - misplaced diacritics,
    - missing vowels,
    - unnecessary spaces inside words.
    
    Output ONLY the corrected Sinhala text. Do NOT add explanations.

    Text:
    {text}
    """

    try:
        corrected = gemini_generate_evaluation(
            prompt,
            budget=EvaluationGeminiClient.OCR_CORRECTION,
            reason="fix_sinhala_ocr",
        ).strip()
        return corrected if corrected else text

    except Exception as e:
        logger.error(f"Error in Sinhala OCR correction: {e}")
        return text

COMBINED_EXAM_PROMPT = """
You are an expert AI for analyzing Sri Lankan exam papers (Sinhala and English medium).
Your task is to extract BOTH the **grading configuration** and the **question structure** into a single strict JSON format.

========================
SECTION 1: CONFIGURATION RULES
========================
Analyze the text to determine how the paper should be graded.
1. **Paper Identification:** Detect if text contains Paper I (Part 1), Paper II (Part 2), or both.
   - Look for headers like "කොටස I" (Part I), "කොටස II" (Part II), "I කොටස", "II කොටස".
   - DO NOT assume Paper I is always MCQ. It can contain structured questions too.
2. **Total Marks:** 
   - Look for "මුළු ලකුණු" (Total Marks) for each section.
   - For Paper I: If MCQs, usually 1 mark each. If structured, look for explicit marks.
   - For Paper II: Look for "Total Marks" or sum the sub-question marks.
3. **Selection Rules:** Read instructions (e.g., "Answer 3 from 5").
   - If "Answer all": {{"mode": "all"}}
   - If "Answer any X": {{"mode": "any", "count": X}}
   - If "Partially compulsory": {{"compulsory": [1, 2], "choose_any": 3}}

========================
SECTION 2: QUESTION STRUCTURE RULES
========================
**Question Detection:**
- Identify main questions (1, 2, 3...) and sub-questions (a, b, c or i, ii, iii).
- Preserve the exact question numbering and mapping from the paper.
- Selection instructions like "answer any 5 from 8" are grading rules only.
  They mean 8 questions are available and 5 are required. NEVER extract only
  the required count. Extract ALL available printed questions.

**MCQ Structure:**
- Detect MCQs by options like (1)(2)(3)(4) or (A)(B)(C)(D).
- If multiple MCQs share a "shared_stem", attach it to the first and use "inherits_shared_stem_from" for others.

**Structured Structure:**
- Use "type": "structured".
- Sub-questions: Map labels (අ, ආ, ඉ or a, b, c) to "sub_questions" dictionary.
- Marks: Extract from "(ලකුණු 05)", "(5 marks)", "20 කි".
- **Rule:** Never assign 0 marks. Use null if unknown.

========================
SECTION 3: ROBUST LAYOUT & TEXT REPAIR
========================
You are receiving text from a system OCR or digital extraction which may have jumbled line orders due to multi-column layouts or PDF encoding issues.
IMPORTANT: If a section says "answer any 05 from 08", that is a requirement rule,
not the number of printed questions. Extract all 8 printed main questions.
1. **Restore Logical Flow**: Use linguistic context to re-order the questions. If sub-question (b) appears before (a) in the text, logically group them correctly in the JSON.
2. **Exhaustive Detection**: Every numbered (1, 2...) or lettered ((a), (b)... / (අ), (ආ)...) item MUST be captured. If sub-questions span multiple pages or are interrupted by tables, bridge the gap and include them all.
3. **Sinhala Repair**: Fix broken conjuncts (ක් ෂ → ක්‍ෂ), diacritic drifts, and accidental spaces within words.

========================
OUTPUT FORMAT (STRICT JSON)
========================
Return a single JSON object where keys are "Paper_I", "Paper_II", "Paper_III", etc. 
Only include keys for papers actually found in the text.

{{
  "Paper_I": {{
    "config": {{
      "subject_detected": "History",
      "medium": "Sinhala", 
      "total_marks": 40,
      "total_questions_available": 9,
      "suggested_weightage": 40,
      "selection_rules": {{"mode": "all"}}
    }},
    "questions": {{
      "1": {{
        "type": "structured",
        "text": "Question text here",
        "marks": 5
      }},
      "2": {{
        "type": "structured",
        "text": "Second question text here",
        "marks": null
      }}
    }}
  }},
  "Paper_III": {{
    "config": {{
      "total_marks": 20,
      "selection_rules": {{"mode": "any", "count": 1}}
    }},
    "questions": {{
      "1": {{ "type": "structured", "text": "Essay prompt...", "marks": 20 }}
    }}
  }}
}}

========================
TEXT TO PROCESS
========================
{content}
"""

def extract_complete_exam_data(text: str):
    """
    Extracts both Configuration (Marks/Rules) and Structure (Questions) in one pass.
    """
    if not text or not text.strip():
        return {}

    try:
        logger.info("Starting combined exam extraction.")

        prepared_text = _prepare_exam_text_for_prompt(text[:30000])
        response_text = gemini_generate(
            COMBINED_EXAM_PROMPT.format(content=prepared_text),
            json_mode=True,
            model_name=settings.EVAL_GEMINI_QUESTION_PARSING_MODEL,
        )

        result = _safe_json_loads(response_text)
        if not result or not isinstance(result, dict):
            return {}

        result = _backfill_missing_segmented_questions(result, prepared_text)
            
        logger.info("Combined exam extraction completed successfully.")
        
        # 🔒 Generic Normalization: Accept any "Paper_*" keys
        cleaned_result = _normalize_extracted_exam_result(result)
        
        # Log basics for debugging
        if cleaned_result.get("Paper_I"):
            logger.info(f"Paper I Detected: {len(cleaned_result['Paper_I'].get('questions', {}))} questions")
        if cleaned_result.get("Paper_II"):
            logger.info(f"Paper II Detected: {len(cleaned_result['Paper_II'].get('questions', {}))} questions")

        return cleaned_result

    except json.JSONDecodeError:
        logger.error("❌ Error: Model output was not valid JSON.")
        return {}

    except Exception as e:
        logger.error(f"❌ Error in combined extraction: {e}")
        return {}

ANSWER_MAPPING_PROMPT = """
You are a STRICT answer extraction AI for Sri Lankan exam papers.
Your ONLY job: find the text the student wrote under each question number. Nothing else.

========================
CRITICAL RULES — READ CAREFULLY
========================
1. **QUESTION NUMBER FIRST**: The ONLY way to assign an answer to a question is if the student wrote that question's number BEFORE the answer text (e.g., "1.", "01.", "1)", "(1)", "1(a)", "2.a", "3", "(අ)", etc.). Look for these number markers in the student's handwriting/OCR.

2. **PAPER ISOLATION**: Sri Lankan papers have distinct parts (Part I, II, III, etc.).
   - Numbering often starts again from 1 or repeats in each part.
   - **CONTEXT HINT**: Look for headers like "භාගය I", "Part II", "III කොටස" in the student text.
   - **IMPORTANT**: If the student text is currently in a section titled "Part III", do NOT map a question "1" found there to a "Part I" question ID.

3. **DO NOT USE TOPIC/MEANING**: NEVER assign an answer based on what the question asks about or what topic the answer covers.

4. **ONE ANSWER PER SECTION**: Each physical section of the student's paper maps to EXACTLY ONE question.

5. **SOURCE MARKER FIELD**: Include the marker you saw if visible. If OCR is noisy but the answer visibly belongs to a question section, include the best marker guess and lower confidence.

6. **OCR CLEANING ONLY**: Fix obvious Sinhala OCR errors. Do NOT correct the student's actual content.

7. **NEVER INFER FROM CONTENT SIMILARITY**: Do not invent answers from the question topic. Only return answer text that appears in the student's OCR.

8. **SUB-QUESTION STRICTNESS**: Do not map a text to `5(අ)`, `5(ආ)`, etc. unless the OCR contains a visible marker such as `5(අ)`, `5 අ`, `5.අ`, `(අ)` under question 5, or another clearly equivalent written marker.

========================
HOW TO FIND QUESTION MARKERS
========================
Look for these common student styles:
- Numeric labels: "1.", "01.", "1)", "(1)", "1-", "2-", "1 ", "2 "
- Sinhala sub labels: "(අ)", "(ආ)", "(ඇ)", "(ඈ)", "අ)", "ආ)", "1. (අ)", "01. (අ)"
- Paper I: "1.", "2.", "3.", "1-", "2-", "(1)", "(2)", "1 ", "2 "
- Paper II: "1(a)", "1.a", "1 a", "1. (අ)", "(අ)", "ආ)", "1.අ"
- Sinhala labels: "1.", "2.", "3." (using Sinhala numerals if applicable, but usually standard digits)

Steps:
1. Scan the student text for Part/Section headers (I, II, III, etc.) to establish context.
2. For each question in the structure, find the corresponding marker in the correct context.
3. If multiple instances of "1" exist, use the "Part" context (e.g., text following a "Part II" header) to decide.
4. Short answers (MCQs) are often just a single word or number (e.g., "1. 3"). Capture these!

========================
INPUTS
========================
QUESTION STRUCTURE (JSON): Question IDs, labels, and the question text.
STUDENT ANSWER TEXT (OCR): Raw OCR text from the student's answer script.

========================
OUTPUT FORMAT (STRICT JSON)
========================
Return raw JSON only in this format:
{{
  "mappings": [
    {{
      "question_id": "exact UUID from structure",
      "label": "visible question/sub-question label, e.g. 2 or 01. (අ)",
      "answer": "cleaned student answer text",
      "source_marker": "visible marker copied from OCR, e.g. 2. or (අ)",
      "confidence": 0.0
    }}
  ]
}}
CRITICAL: `question_id` MUST BE the exact `id` string (UUID) from the structure.
ONLY include keys for questions that the student actually ATTEMPTED.
CRITICAL AVOID MAPPING QUESTION TEXT: If a section only contains the phrasing of the question itself, do NOT map it.
For compatibility, never wrap the JSON in Markdown and never add commentary.
Return compact minified JSON, not pretty-printed JSON, to avoid truncation.

========================
QUESTION STRUCTURE
========================
{structure}

========================
STUDENT ANSWER TEXT
========================
{answer_text}
"""

PAPER_II_MAIN_BLOCK_PROMPT = """
You are a STRICT answer extraction AI for Sri Lankan exam papers.
You are given ONE already-isolated Paper II answer block for a single main question number.

Your job:
- map the student's writing inside this one block to the given sub-question IDs only
- return ONLY text that visibly appears in this block
- do NOT use topic matching from outside this block

Rules:
1. The OCR block already belongs to main question {main_no}. Do NOT map outside this main question.
2. Prefer visible sub-question markers like `(අ)`, `(ආ)`, `(ඇ)`, `අ)`, `ආ)`, `a`, `b`, `c`.
3. If inner sub-question markers are weak or missing, but the student clearly wrote multiple answers in sequence inside this same main-question block, split them in the same order as the provided structure.
4. If the block only supports one sub-question answer confidently, return only that one.
5. Copy the student's wording from the block. Only do light OCR cleanup.
6. Do NOT invent missing answers.

Return raw JSON only:
{{"uuid": "student answer text"}}

MAIN QUESTION NUMBER:
{main_no}

SUB-QUESTION STRUCTURE:
{structure}

OCR BLOCK:
{answer_text}
"""


def _is_hallucinated_question_text(mapped_text: str, q_text: str, threshold: float = 0.85) -> bool:
    """
    Detects if the mapped student answer is just a hallucination of the
    original printed question text from the paper.
    """
    if not q_text or not mapped_text or len(mapped_text.strip()) < 5:
        return False

    q_norm = q_text.strip().replace(" ", "")
    m_norm = mapped_text.strip().replace(" ", "")

    # Basic length check: if the student provided a very long answer,
    # it's likely not just a hallucinated question even if it contains the question.
    if len(m_norm) > len(q_norm) + 50:
        return False

    # Check exact match or very high similarity
    ratio = difflib.SequenceMatcher(None, q_norm[:len(m_norm)], m_norm).ratio()
    if ratio > threshold:
        return True

    # Check if the mapping is a substring of the question (often happens with OCR fragments)
    if len(m_norm) > 10 and m_norm in q_norm:
        return True

    return False


def _normalize_marker_text(text: str) -> str:
    # New helper: normalize OCR-heavy text before checking whether
    # a mapped snippet is actually anchored to a visible question marker.
    normalized = re.sub(r'\s+', ' ', str(text or "").lower()).strip()
    return re.sub(r'[\s\.,;:!?\-()\[\]{}"\'`]+', '', normalized)


def _normalize_ocr_for_matching(text: str) -> str:
    """Collapse OCR spacing/punctuation while keeping Sinhala glyph order."""
    normalized = re.sub(r'\s+', ' ', str(text or "").lower()).strip()
    return re.sub(r'[\s\.,;:!?\-()\[\]{}"\'`/\\|]+', '', normalized)


def _answer_text_supported_by_source(source_text: str, mapped_text: str) -> bool:
    """
    Tolerant answer-presence check. Exact full-snippet matching is too brittle
    for Sinhala OCR, so accept substantial fuzzy overlap with the source text.
    """
    source_norm = _normalize_ocr_for_matching(source_text)
    mapped_norm = _normalize_ocr_for_matching(mapped_text)
    if not source_norm or not mapped_norm:
        return False

    if mapped_norm in source_norm:
        return True

    if len(mapped_norm) < 8:
        return False

    # Compare against same-sized windows to tolerate a few OCR substitutions.
    step = max(6, len(mapped_norm) // 4)
    best_ratio = 0.0
    window_size = min(len(source_norm), max(len(mapped_norm) + 20, int(len(mapped_norm) * 1.35)))
    for start in range(0, max(1, len(source_norm) - window_size + 1), step):
        window = source_norm[start:start + window_size]
        best_ratio = max(best_ratio, difflib.SequenceMatcher(None, mapped_norm, window).ratio())
        if best_ratio >= 0.72:
            return True

    # Sinhala character n-gram overlap handles missing spaces and minor spelling noise.
    if len(mapped_norm) >= 12:
        grams = {mapped_norm[i:i + 3] for i in range(0, len(mapped_norm) - 2)}
        if grams:
            hits = sum(1 for gram in grams if gram in source_norm)
            if hits / len(grams) >= 0.62:
                return True

    return False


def _strip_answer_marker(answer: str, label: str = "", source_marker: str = "") -> str:
    text = str(answer or "").strip()
    markers = [source_marker, label]
    for marker in markers:
        marker = str(marker or "").strip()
        if not marker:
            continue
        pattern = rf'^\s*[\(\[]?\s*{re.escape(marker)}\s*[\)\]\.\-:]*\s*'
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    return text


def _coerce_answer_mapping(parsed) -> dict[str, dict]:
    """
    Normalize old and new Gemini mapping schemas into structured entries.
    Returned entries still collapse to strings before persistence, preserving
    grading compatibility.
    """
    entries: list[dict] = []

    if isinstance(parsed, dict) and isinstance(parsed.get("mappings"), list):
        entries = parsed.get("mappings") or []
    elif isinstance(parsed, list):
        entries = parsed
    elif isinstance(parsed, dict):
        for key, value in parsed.items():
            if key == "mappings":
                continue
            if isinstance(value, dict):
                entry = dict(value)
                entry.setdefault("question_id", key)
                entries.append(entry)
            else:
                entries.append({"question_id": key, "answer": value})

    normalized: dict[str, dict] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        qid = str(entry.get("question_id") or entry.get("id") or "").strip()
        if not qid:
            continue
        answer = entry.get("answer")
        if answer is None:
            answer = entry.get("student_answer")
        if answer is None:
            answer = entry.get("text")
        if answer is None:
            continue
        answer_text = str(answer).strip()
        if not answer_text or answer_text.lower() in {"null", "none"}:
            continue
        confidence = entry.get("confidence", 0.0)
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = 0.0
        normalized[qid] = {
            "question_id": qid,
            "label": str(entry.get("label") or "").strip(),
            "answer": answer_text,
            "source_marker": str(entry.get("source_marker") or entry.get("marker") or "").strip(),
            "confidence": confidence,
        }
    return normalized


def _normalize_answer_label_for_lookup(label: str, *, strip_leading_zeroes: bool = True) -> str:
    text = str(label or "").strip().lower()
    if strip_leading_zeroes:
        text = re.sub(r'^\s*0+(\d)', r'\1', text)
    text = re.sub(r'\s+', '', text)
    text = text.replace("[", "(").replace("]", ")")
    text = re.sub(r'[\.\-:]+', '', text)
    text = re.sub(r'^\((\d+)\)$', r'\1', text)
    return text


def _build_unique_label_lookup(flat_structure: list[dict]) -> tuple[dict[str, str], dict[str, str]]:
    exact_buckets: dict[str, list[str]] = {}
    buckets: dict[str, list[str]] = {}
    for item in flat_structure or []:
        qid = str(item.get("id") or "")
        label = str(item.get("label") or "")
        if not qid or not label:
            continue
        exact_normalized = _normalize_answer_label_for_lookup(label, strip_leading_zeroes=False)
        if exact_normalized:
            exact_buckets.setdefault(exact_normalized, []).append(qid)
        variants = {
            label,
            label.lstrip("0"),
            re.sub(r'^0+(\d+)', r'\1', label),
        }
        for variant in variants:
            normalized = _normalize_answer_label_for_lookup(variant)
            if normalized:
                buckets.setdefault(normalized, []).append(qid)

    exact_lookup = {
        label: ids[0]
        for label, ids in exact_buckets.items()
        if len(set(ids)) == 1
    }
    loose_lookup = {
        label: ids[0]
        for label, ids in buckets.items()
        if len(set(ids)) == 1
    }
    return exact_lookup, loose_lookup


def _resolve_mapping_question_id(
    entry: dict,
    by_id: dict,
    exact_label_lookup: dict[str, str],
    loose_label_lookup: dict[str, str],
) -> tuple[str, str]:
    qid = str(entry.get("question_id") or "").strip()
    if qid in by_id:
        return qid, "id"

    label = str(entry.get("label") or entry.get("source_marker") or "").strip()
    exact_label = _normalize_answer_label_for_lookup(label, strip_leading_zeroes=False)
    if exact_label and exact_label in exact_label_lookup:
        return exact_label_lookup[exact_label], "label"

    normalized_label = _normalize_answer_label_for_lookup(label, strip_leading_zeroes=True)
    if normalized_label and normalized_label in loose_label_lookup:
        return loose_label_lookup[normalized_label], "label"

    return qid, "invalid_id"


def _build_label_marker_patterns(label: str) -> list[str]:
    # New helper: build tolerant regex patterns for the label styles students
    # commonly write, so we validate mappings against markers, not semantics.
    raw_label = str(label or "").strip()
    if not raw_label:
        return []

    patterns = []
    normalized = _normalize_marker_text(raw_label)
    match = re.match(r'^\s*0*(\d+)(?:\s*[\.\-:]?\s*\(?\s*([^\)]+?)\s*\)?)?\s*$', raw_label)

    if match:
        main_no, sub_label = match.groups()
        main_variants = {main_no, main_no.zfill(2)}
        if sub_label:
            sub_clean = re.escape(sub_label.strip())
            for main_variant in sorted(main_variants, key=len, reverse=True):
                escaped_main = re.escape(main_variant)
                patterns.extend([
                    rf'(?<!\d){escaped_main}\s*[\.\-:)]?\s*\(\s*{sub_clean}\s*\)',
                    rf'(?<!\d){escaped_main}\s*[\.\-:)]?\s*{sub_clean}(?!\w)',
                    rf'(?:(?<=\n)|(?<=\r)|^)\s*{escaped_main}\s*[\.\-:)]\s*\(\s*{sub_clean}\s*\)',
                ])
            patterns.extend([
                rf'(?:(?<=\n)|(?<=\r)|^)\s*\(\s*{sub_clean}\s*\)',
                rf'(?:(?<=\n)|(?<=\r)|^)\s*{sub_clean}[\)\.\-:]',
            ])
        else:
            for main_variant in sorted(main_variants, key=len, reverse=True):
                escaped_main = re.escape(main_variant)
                patterns.extend([
                    rf'(?:(?<=\n)|(?<=\r)|^)\s*\(?\s*{escaped_main}\s*[\)\.\-:\s]',
                    rf'(?<!\d)\(\s*{escaped_main}\s*\)',
                    rf'(?<!\d){escaped_main}\s*\)',
                ])

    if normalized:
        patterns.append(re.escape(normalized))

    # Preserve order while removing duplicates.
    seen = set()
    unique_patterns = []
    for pattern in patterns:
        if pattern not in seen:
            seen.add(pattern)
            unique_patterns.append(pattern)
    return unique_patterns


def _extract_main_and_sub_label(label: str) -> tuple[str, str]:
    # New helper: split a label like `2(අ)` into its main-question number and
    # sub-question marker so we can validate nearby OCR context more flexibly.
    raw_label = str(label or "").strip()
    match = re.match(r'^\s*0*(\d+)(?:\s*[\.\-:]?\s*\(?\s*([^\)]+?)\s*\)?)?\s*$', raw_label)
    if not match:
        return "", ""
    main_no, sub_label = match.groups()
    return (main_no or "").strip(), (sub_label or "").strip()


def _looks_like_main_number_context(window_text: str, main_no: str) -> bool:
    # New helper: accept OCR variants of the main question number as long as the
    # nearby text still looks like a numbered answer block.
    if not window_text or not main_no:
        return False

    patterns = [
        rf'(?:(?<=\n)|(?<=\r)|^)\s*{re.escape(main_no)}[\)\.\-:\s]',
        rf'(?<!\d){re.escape(main_no)}\s*\)',
        rf'(?<!\d){re.escape(main_no)}\s*[\.\-:]\s*',
    ]
    return any(re.search(pattern, window_text, flags=re.IGNORECASE | re.MULTILINE) for pattern in patterns)


def _build_sub_label_variants(sub_label: str) -> list[str]:
    # New helper: OCR can distort Sinhala sub-labels, so we tolerate the most
    # common nearby variants when checking whether the student wrote `(අ)/(ආ)/...`.
    raw = str(sub_label or "").strip()
    if not raw:
        return []

    variants = {raw}

    # Keep only Sinhala letters for OCR-tolerant matching.
    sinhala_only = "".join(ch for ch in raw if '\u0D80' <= ch <= '\u0DFF')
    if sinhala_only:
        variants.add(sinhala_only)
        variants.add(sinhala_only.replace("ැ", "ැ"))

    # Frequent OCR confusion in the current data: `(ඈඇ)` may appear as just
    # one of the visible glyphs, so keep the individual characters too.
    if len(sinhala_only) > 1:
        for ch in sinhala_only:
            variants.add(ch)

    return [v for v in variants if v]


def _has_nearby_subquestion_support(source_text: str, mapped_text: str, label: str) -> bool:
    """
    New fallback validator for Paper II sub-questions:
    if OCR weakens the exact sub-question marker, accept the mapping when the
    answer snippet appears near the right main-question number and a plausible
    nearby sub-label variant.
    """
    if not source_text or not mapped_text or not label:
        return False

    main_no, sub_label = _extract_main_and_sub_label(label)
    if not main_no or not sub_label:
        return False

    source = str(source_text)
    mapped_norm = _normalize_marker_text(mapped_text)
    sub_variants = _build_sub_label_variants(sub_label)
    if not mapped_norm or not sub_variants:
        return False

    main_patterns = [
        rf'(?:(?<=\n)|(?<=\r)|^)\s*{re.escape(main_no)}[\)\.\-:\s]',
        rf'(?<!\d){re.escape(main_no)}\s*\)',
    ]

    for pattern in main_patterns:
        for match in re.finditer(pattern, source, flags=re.IGNORECASE | re.MULTILINE):
            # New window strategy: look at the answer block immediately after the
            # main question marker instead of requiring an exact `2(අ)` style hit.
            window = source[match.start():match.start() + 1400]
            normalized_window = _normalize_marker_text(window)
            if mapped_norm not in normalized_window:
                continue

            if not _looks_like_main_number_context(window, main_no):
                continue

            if any(variant in normalized_window for variant in [_normalize_marker_text(v) for v in sub_variants]):
                return True

    return False


def _has_direct_marker_supported_mapping(source_text: str, mapped_text: str, label: str) -> bool:
    """
    New helper: keep the stricter marker checks separate from the relaxed
    main-block fallback so Paper II answers can be reassigned more safely later.
    """
    if not source_text or not mapped_text or not label:
        return False

    source = str(source_text)
    mapped_norm = _normalize_marker_text(mapped_text)
    if len(mapped_norm) < 4:
        return False

    marker_patterns = _build_label_marker_patterns(label)
    if not marker_patterns:
        return False

    normalized_source = _normalize_marker_text(source)

    for pattern in marker_patterns:
        if pattern == re.escape(_normalize_marker_text(label)):
            marker_match = re.search(pattern, normalized_source, flags=re.IGNORECASE)
            if not marker_match:
                continue
            start_idx = marker_match.start()
            window = normalized_source[start_idx:start_idx + 900]
            if mapped_norm in window:
                return True
            if _answer_text_supported_by_source(source, mapped_text):
                return True
            continue

        for match in re.finditer(pattern, source, flags=re.IGNORECASE | re.MULTILINE):
            window = source[match.start():match.start() + 1400]
            if mapped_norm in _normalize_marker_text(window):
                return True
            if _answer_text_supported_by_source(window, mapped_text):
                return True

    return False


def _has_main_question_block_support(source_text: str, mapped_text: str, label: str) -> bool:
    """
    New fallback for handwritten Paper II answers: OCR often keeps the main
    question number but drops the inner `(අ)/(ආ)/(ඇ)` marker. In that case,
    accept the mapping if the full answer text clearly appears inside the
    numbered answer block for the correct main question.
    """
    if not source_text or not mapped_text or not label:
        return False

    main_no, sub_label = _extract_main_and_sub_label(label)
    if not main_no or not sub_label:
        return False

    mapped_text = str(mapped_text or "").strip()
    mapped_norm = _normalize_marker_text(mapped_text)
    if len(mapped_norm) < 20:
        return False

    # New safety gate: only relax the rule for substantial answer-like text so
    # we do not reintroduce the old short-snippet false mappings.
    if not _looks_like_full_main_answer(mapped_text):
        return False

    source = str(source_text)
    main_patterns = [
        rf'(?:(?<=\n)|(?<=\r)|^)\s*{re.escape(main_no)}[\)\.\-:\s]',
        rf'(?<!\d){re.escape(main_no)}\s*\)',
        rf'(?<!\d){re.escape(main_no)}\s*[\.\-:]\s*',
    ]

    numbered_answer_pattern = re.compile(
        r'(?:(?<=\n)|(?<=\r)|^)\s*\d+\s*[\)\.\-:\s]',
        flags=re.IGNORECASE | re.MULTILINE,
    )

    seen_starts = set()
    for pattern in main_patterns:
        for match in re.finditer(pattern, source, flags=re.IGNORECASE | re.MULTILINE):
            if match.start() in seen_starts:
                continue
            seen_starts.add(match.start())

            block_start = match.start()
            tail = source[block_start + 1:]
            next_match = numbered_answer_pattern.search(tail)
            block_end = len(source)
            if next_match:
                block_end = block_start + 1 + next_match.start()

            block = source[block_start:block_end]
            if mapped_norm in _normalize_marker_text(block):
                return True
            if _answer_text_supported_by_source(block, mapped_text):
                return True

    return False


def _extract_main_question_block(source_text: str, main_no: str) -> tuple[str, int]:
    """
    New helper: isolate the OCR block that begins at a Paper II main question
    number so deferred sub-question snippets can be reassigned by position.
    """
    if not source_text or not main_no:
        return "", -1

    source = str(source_text)
    main_patterns = [
        rf'(?:(?<=\n)|(?<=\r)|^)\s*{re.escape(main_no)}[\)\.\-:\s]',
        rf'(?<!\d){re.escape(main_no)}\s*\)',
        rf'(?<!\d){re.escape(main_no)}\s*[\.\-:]\s*',
    ]
    numbered_answer_pattern = re.compile(
        r'(?:(?<=\n)|(?<=\r)|^)\s*\d+\s*[\)\.\-:\s]',
        flags=re.IGNORECASE | re.MULTILINE,
    )

    best_match = None
    for pattern in main_patterns:
        match = re.search(pattern, source, flags=re.IGNORECASE | re.MULTILINE)
        if match and (best_match is None or match.start() < best_match.start()):
            best_match = match

    if not best_match:
        return "", -1

    block_start = best_match.start()
    tail = source[block_start + 1:]
    next_match = numbered_answer_pattern.search(tail)
    block_end = len(source)
    if next_match:
        block_end = block_start + 1 + next_match.start()

    return source[block_start:block_end], block_start


def _find_normalized_substring_index(haystack: str, needle: str) -> int:
    """
    New helper: find OCR answer snippets after normalization so ordering can be
    estimated even when spacing and punctuation drift.
    """
    haystack_norm = _normalize_marker_text(haystack)
    needle_norm = _normalize_marker_text(needle)
    if not haystack_norm or not needle_norm:
        return -1
    return haystack_norm.find(needle_norm)


def _split_answer_text_by_number_restart(answer_text: str) -> tuple[str, str]:
    """
    New helper: answer sheets usually restart numbering from `1` when Paper II
    begins. Split the OCR text at the last visible restart so Paper I and
    Paper II validators do not compete over the same repeated numbers.
    """
    text = str(answer_text or "")
    restart_pattern = re.compile(
        r'(?:(?<=\n)|(?<=\r)|^|\s)0*1\s*[\)\.\-:]',
        flags=re.IGNORECASE | re.MULTILINE,
    )
    matches = list(restart_pattern.finditer(text))
    if len(matches) < 2:
        return text, text

    restart_at = matches[-1].start()
    return text[:restart_at], text[restart_at:]


def _get_part_specific_answer_text(answer_text: str, part_name: str) -> str:
    """
    New helper: isolate repeated numbering scopes so Paper II matching uses the
    OCR tail where long answers restart from `1`.
    """
    paper_i_text, paper_ii_text = _split_answer_text_by_number_restart(answer_text)
    normalized_part = str(part_name or "").strip().lower()

    if "paper_ii" in normalized_part or "part ii" in normalized_part:
        return paper_ii_text or str(answer_text or "")
    if "paper_i" in normalized_part or "part i" in normalized_part:
        return paper_i_text or str(answer_text or "")
    return str(answer_text or "")


def _has_marker_supported_mapping(source_text: str, mapped_text: str, label: str) -> bool:
    """
    New validation layer: keep Gemini in an extraction role by requiring
    source-text marker support for the returned answer snippet.
    """
    if not source_text or not mapped_text or not label:
        return False

    # New early decision: keep direct marker checks separate from relaxed
    # main-block support so Paper II reassignment can decide how to use each.
    if _has_direct_marker_supported_mapping(source_text, mapped_text, label):
        return True

    if _has_nearby_subquestion_support(source_text, mapped_text, label):
        return True

    if _has_main_question_block_support(source_text, mapped_text, label):
        return True

    return False

    source = str(source_text)
    mapped_norm = _normalize_marker_text(mapped_text)
    if len(mapped_norm) < 4:
        return False

    marker_patterns = _build_label_marker_patterns(label)
    if not marker_patterns:
        return False

    normalized_source = _normalize_marker_text(source)

    for pattern in marker_patterns:
        if pattern == re.escape(_normalize_marker_text(label)):
            marker_match = re.search(pattern, normalized_source, flags=re.IGNORECASE)
            if not marker_match:
                continue
            start_idx = marker_match.start()
            window = normalized_source[start_idx:start_idx + 900]
            if mapped_norm in window:
                return True
            continue

        for match in re.finditer(pattern, source, flags=re.IGNORECASE | re.MULTILINE):
            window = source[match.start():match.start() + 900]
            if mapped_norm in _normalize_marker_text(window):
                return True

    # New fallback for Paper II style sub-questions where OCR often weakens the
    # exact `(අ)/(ආ)/(ඈඇ)` marker but still preserves nearby main-question context.
    if _has_nearby_subquestion_support(source_text, mapped_text, label):
        return True

    # New final fallback: some answer sheets preserve only the main question
    # number, so let full-length Paper II sub-answers pass when they clearly
    # belong to the correct numbered answer block.
    if _has_main_question_block_support(source_text, mapped_text, label):
        return True

    return False


def _clean_local_answer_block(text: str) -> str:
    raw = str(text or "")
    if re.match(
        r'^\s*[-–—]\s*(?:$|(?:\d+\s*)?කොටස\b|[ivx]+\s*කොටස\b|part\b)',
        raw,
        flags=re.IGNORECASE,
    ):
        return ""
    text = re.sub(r'---\s*PAGE\s+\d+\s*---', ' ', raw, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    cleaned = text.strip(" .:-–—")
    return "" if _is_invalid_mapped_answer(cleaned) else cleaned


def _is_invalid_mapped_answer(value: str) -> bool:
    """Reject placeholders and section headers that should not be graded."""
    text = re.sub(r'\s+', ' ', str(value or "")).strip()
    if not text:
        return True
    if text.lower() in {"null", "none", "n/a", "na"}:
        return True
    if re.fullmatch(r'[-–—_.\s]+', text):
        return True
    if re.fullmatch(r'[\(?\[]?\s*[-–—]\s*[\)?\]]?', text):
        return True
    normalized = _normalize_marker_text(text)
    if len(normalized) < 2:
        return True
    header_patterns = [
        r'^\s*(?:\d+\s*)?කොටස\s*[-–—:]?\s*',
        r'^\s*(?:i|ii|iii|iv|v|1|2)\s*කොටස\s*[-–—:]?\s*',
        r'^\s*විග්',
        r'^\s*කෙටි\s+පිළිතුරු',
        r'^\s*short\s+answers?\b',
        r'^\s*essay\s+questions?\b',
    ]
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in header_patterns)


def _looks_like_sub_marker_token(token: str) -> bool:
    token = str(token or "").strip()
    if not token or len(token) > 4:
        return False
    if re.search(r'\d', token):
        return False
    return bool(re.search(r'[A-Za-z\u0D80-\u0DFF]', token))


def _find_long_answer_start(answer_text: str) -> int:
    text = str(answer_text or "")
    patterns = [
        r'(?<!\d)0[1-9]\s*[\.\)]\s*\(\s*[\u0D80-\u0DFFA-Za-z]{1,4}\s*\)',
        r'(?<!\d)0[1-9]\s*[\.\)]\s*[A-Za-z]\b',
    ]
    starts = []
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            starts.append(match.start())
    return min(starts) if starts else -1


def _extract_numeric_answer_blocks(text: str) -> dict[str, str]:
    source = str(text or "")
    marker_pattern = re.compile(r'(?<!\d)(0?[1-9]|[12]\d)\s*[\.\)]\s+', flags=re.IGNORECASE)
    matches = list(marker_pattern.finditer(source))
    blocks: dict[str, str] = {}

    for index, match in enumerate(matches):
        label = str(int(match.group(1)))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        value = _clean_local_answer_block(source[start:end])
        if value:
            blocks[label] = value

    return blocks


def _extract_long_answer_sub_blocks(text: str) -> dict[str, list[str]]:
    source = str(text or "")
    main_pattern = re.compile(
        r'(?<!\d)0*([1-9]|[12]\d)\s*[\.\)]\s*\(\s*([^)]+?)\s*\)',
        flags=re.IGNORECASE,
    )
    main_matches = [
        match for match in main_pattern.finditer(source)
        if _looks_like_sub_marker_token(match.group(2))
    ]
    grouped: dict[str, list[str]] = {}

    for index, main_match in enumerate(main_matches):
        main_no = str(int(main_match.group(1)))
        block_end = main_matches[index + 1].start() if index + 1 < len(main_matches) else len(source)
        block = source[main_match.start():block_end]
        sub_pattern = re.compile(r'\(\s*([^)]+?)\s*\)', flags=re.IGNORECASE)
        sub_matches = [
            match for match in sub_pattern.finditer(block)
            if _looks_like_sub_marker_token(match.group(1))
        ]
        if not sub_matches:
            continue

        values = []
        for sub_index, sub_match in enumerate(sub_matches):
            start = sub_match.end()
            end = sub_matches[sub_index + 1].start() if sub_index + 1 < len(sub_matches) else len(block)
            value = _clean_local_answer_block(block[start:end])
            # Preserve blank sub-parts as placeholders so later answers do not
            # shift into the wrong label, e.g. `(අ) answer (ආ) - (ඇ) answer`.
            values.append(value)
        if any(value for value in values):
            grouped[main_no] = values

    return grouped


def _map_answers_from_visible_ocr_markers(
    answer_text: str,
    flat_structure: list[dict],
    existing: dict,
    *,
    overwrite_existing: bool = False,
) -> dict:
    """
    Deterministic fallback for when Gemini skips a chunk. It maps visible OCR
    answer blocks by written markers only, so it fills gaps without semantic
    guessing.
    """
    recovered: dict[str, str] = {}
    existing = existing or {}
    text = str(answer_text or "")
    long_start = _find_long_answer_start(text)
    short_text = text[:long_start] if long_start >= 0 else text
    long_text = text[long_start:] if long_start >= 0 else ""

    short_blocks = _extract_numeric_answer_blocks(short_text)
    long_blocks = _extract_long_answer_sub_blocks(long_text)

    short_items: dict[str, dict] = {}
    sub_items_by_main: dict[str, list[dict]] = {}
    for item in flat_structure or []:
        qid = str(item.get("id") or "")
        if not qid or (qid in existing and not overwrite_existing):
            continue
        if item.get("is_sub_question"):
            main_no = str(item.get("parent_main_label") or "").lstrip("0")
            if main_no:
                sub_items_by_main.setdefault(main_no, []).append(item)
            continue
        if item.get("has_sub_questions"):
            continue
        label = str(item.get("label") or "").strip()
        if re.fullmatch(r'0*\d+', label):
            short_items[str(int(label))] = item

    for label, value in short_blocks.items():
        item = short_items.get(label)
        if not item:
            continue
        qid = str(item.get("id"))
        if (qid not in existing or overwrite_existing) and value and not _is_invalid_mapped_answer(value):
            recovered[qid] = value

    for main_no, values in long_blocks.items():
        items = sub_items_by_main.get(main_no) or []
        # Preserve question-paper order. Sinhala OCR often corrupts sub-labels,
        # but the answer script usually keeps the written sub-parts in sequence.
        for item, value in zip(items, values):
            qid = str(item.get("id"))
            if (qid not in existing or overwrite_existing) and value and not _is_invalid_mapped_answer(value):
                recovered[qid] = value

    return recovered


def _count_visible_ocr_answer_blocks(answer_text: str) -> int:
    text = str(answer_text or "")
    long_start = _find_long_answer_start(text)
    short_text = text[:long_start] if long_start >= 0 else text
    long_text = text[long_start:] if long_start >= 0 else ""
    short_count = len(_extract_numeric_answer_blocks(short_text))
    long_count = sum(
        1
        for values in _extract_long_answer_sub_blocks(long_text).values()
        for value in values
        if value
    )
    return short_count + long_count


def _cleanup_final_answer_mappings(mappings: dict, flat_structure: list[dict]) -> dict:
    """Final guardrail before persistence: remove placeholders, parent duplicates,
    and repeated sub-part answers caused by model label drift."""
    if not mappings:
        return {}

    item_by_id = {str(item.get("id")): item for item in flat_structure or []}
    children_by_parent: dict[tuple[str, str], set[str]] = {}
    for item in flat_structure or []:
        if not item.get("is_sub_question"):
            continue
        parent_key = (
            str(item.get("part") or ""),
            str(item.get("parent_main_label") or "").lstrip("0"),
        )
        children_by_parent.setdefault(parent_key, set()).add(str(item.get("id")))

    cleaned: dict[str, str] = {}
    for key, value in mappings.items():
        value_text = str(value or "").strip()
        if _is_invalid_mapped_answer(value_text):
            logger.warning("Dropping invalid/placeholder mapped answer for ID %s: %s", key, value_text[:80])
            continue
        cleaned[str(key)] = value_text

    # If a main question has visible mapped sub-parts, keep the sub-parts and
    # remove the parent display mapping. Grading already operates on leaves.
    for key, value in list(cleaned.items()):
        item = item_by_id.get(str(key), {})
        if not item.get("has_sub_questions"):
            continue
        parent_key = (
            str(item.get("part") or ""),
            str(item.get("parent_main_label") or item.get("label") or "").lstrip("0"),
        )
        child_ids = children_by_parent.get(parent_key, set())
        if any(child_id in cleaned for child_id in child_ids):
            logger.warning("Dropping parent mapping for ID %s because child sub-parts are mapped.", key)
            cleaned.pop(key, None)

    # Within one Paper II main question, the same normalized answer should not
    # appear for multiple sub-parts. Keep the first occurrence in paper order.
    seen_by_main: dict[tuple[str, str], set[str]] = {}
    ordered_keys = sorted(
        cleaned.keys(),
        key=lambda key: next(
            (idx for idx, item in enumerate(flat_structure or []) if str(item.get("id")) == str(key)),
            10**9,
        ),
    )
    for key in ordered_keys:
        item = item_by_id.get(str(key), {})
        if not item.get("is_sub_question"):
            continue
        main_key = (
            str(item.get("part") or ""),
            str(item.get("parent_main_label") or "").lstrip("0"),
        )
        value_norm = _normalize_marker_text(cleaned.get(key, ""))
        if len(value_norm) < 8:
            continue
        seen = seen_by_main.setdefault(main_key, set())
        if value_norm in seen:
            logger.warning(
                "Dropping duplicate sub-part mapping for ID %s under main %s.",
                key,
                main_key[1],
            )
            cleaned.pop(key, None)
            continue
        seen.add(value_norm)

    return cleaned


def _looks_like_full_main_answer(mapped_text: str) -> bool:
    # New heuristic: main-question answers should look materially longer than
    # the short snippets that often belong to Paper I objective answers.
    text = str(mapped_text or "").strip()
    normalized = _normalize_marker_text(text)
    if len(normalized) >= 80:
        return True
    if len(re.findall(r'\S+', text)) >= 12:
        return True
    if len(re.findall(r'[.!?]|[।]|[\n\r]', text)) >= 2:
        return True
    if re.search(r'[\(\[]\s*[අආඇඈඉEeAaBbCcIiVv]+\s*[\)\]]', text):
        return True
    return False


def _suppress_cross_part_duplicate_mappings(all_mappings: dict, flat_structure: list) -> dict:
    """
    New safeguard: prevent the same short snippet from counting as both a
    Paper I answer and a Paper II main-question answer.
    """
    if not all_mappings:
        return all_mappings

    item_by_id = {str(item.get("id")): item for item in flat_structure}
    grouped: dict[str, list[tuple[str, str, dict]]] = {}

    for key, value in all_mappings.items():
        norm_text = _normalize_marker_text(value)
        if len(norm_text) < 4:
            continue
        grouped.setdefault(norm_text, []).append((key, value, item_by_id.get(str(key), {})))

    filtered = dict(all_mappings)
    for _, entries in grouped.items():
        if len(entries) < 2:
            continue

        keepers = []
        for key, value, item in entries:
            is_main_with_children = bool(item.get("has_sub_questions"))
            if is_main_with_children and not _looks_like_full_main_answer(value):
                logger.warning(
                    "Discarding duplicate short mapping for ID %s - looks like a short-answer snippet, not a full main-question answer.",
                    key,
                )
                filtered.pop(key, None)
                continue
            keepers.append((key, value, item))

        if keepers:
            continue

    return filtered


def _is_paper_ii_part(part_name: str) -> bool:
    # New helper: keep Paper II-specific recovery logic scoped to the long-answer
    # section where numbering restarts and sub-question markers are less reliable.
    normalized = str(part_name or "").strip().lower().replace(" ", "_")
    return normalized in {"paper_ii", "part_ii"} or "paper_ii" in normalized or "part_ii" in normalized


def _map_paper_ii_main_block_subparts(block_text: str, main_no: str, subparts: list[dict]) -> dict:
    # New helper: after isolating one numbered Paper II answer block, ask Gemini
    # to split only that block into the relevant sub-parts instead of guessing
    # sub-question labels from the whole answer script at once.
    if not block_text or not main_no or not subparts:
        return {}

    try:
        response_text = gemini_generate(
            PAPER_II_MAIN_BLOCK_PROMPT.format(
                main_no=main_no,
                structure=json.dumps(subparts, ensure_ascii=False, indent=2),
                answer_text=str(block_text)[:12000],
            ),
            json_mode=True,
        )
        parsed = _safe_json_loads(response_text) if response_text else {}
        if not isinstance(parsed, dict):
            return {}
        return parsed
    except Exception as e:
        logger.warning("Paper II main-block mapping failed for main %s: %s", main_no, e)
        return {}


def _reassign_paper_ii_main_block_candidates(
    answer_text: str,
    deferred_candidates: list[dict],
    chunk: list[dict],
    filtered_result: dict,
) -> None:
    """
    New post-pass for Paper II: when OCR only proves the main-question block,
    reassign the returned snippets to that main question's sub-parts in block
    order instead of trusting Gemini's guessed sub-label directly.
    """
    if not deferred_candidates:
        return

    deferred_by_main: dict[str, list[dict]] = {}
    for candidate in deferred_candidates:
        deferred_by_main.setdefault(candidate["main_no"], []).append(candidate)

    chunk_by_main: dict[str, list[dict]] = {}
    for item in chunk:
        if not item.get("is_sub_question"):
            continue
        main_no = str(item.get("parent_main_label") or "")
        chunk_by_main.setdefault(main_no, []).append(item)

    for main_no, candidates in deferred_by_main.items():
        block_text, _ = _extract_main_question_block(answer_text, main_no)
        if not block_text:
            continue

        ordered_subparts = chunk_by_main.get(main_no, [])
        if not ordered_subparts:
            continue

        accepted_ids = set(filtered_result.keys())
        remaining_subparts = [item for item in ordered_subparts if item["id"] not in accepted_ids]
        if not remaining_subparts:
            continue

        unique_candidates = {}
        for candidate in candidates:
            unique_candidates.setdefault(_normalize_marker_text(candidate["text"]), candidate)

        sorted_candidates = sorted(
            unique_candidates.values(),
            key=lambda candidate: (
                _find_normalized_substring_index(block_text, candidate["text"])
                if _find_normalized_substring_index(block_text, candidate["text"]) >= 0
                else 10**9
            ),
        )

        for sub_item, candidate in zip(remaining_subparts, sorted_candidates):
            filtered_result[sub_item["id"]] = candidate["text"]
            logger.info(
                "Reassigned deferred Paper II main-block answer from label %s to %s.",
                candidate.get("original_label"),
                sub_item.get("label"),
            )


def _fill_paper_ii_subparts_from_main_blocks(
    answer_text: str,
    chunk: list[dict],
    filtered_result: dict,
    allowed_main_nos: set[str] | None = None,
) -> None:
    """
    New recovery pass for Paper II:
    once a main-question block is isolated by number, map only that block to the
    remaining sub-parts so we do not depend on whole-paper leaf-label guesses.
    """
    subparts_by_main: dict[str, list[dict]] = {}
    for item in chunk:
        if not item.get("is_sub_question"):
            continue
        main_no = str(item.get("parent_main_label") or "")
        if not main_no:
            continue
        subparts_by_main.setdefault(main_no, []).append(item)

    for main_no, subparts in subparts_by_main.items():
        # New safety gate: only expand mains that already showed evidence in the
        # first whole-paper pass, so recovery does not invent extra attempted
        # mains like 3/6/7 just because a numbered block exists somewhere.
        if allowed_main_nos is not None and main_no not in allowed_main_nos:
            continue

        remaining_subparts = [item for item in subparts if item["id"] not in filtered_result]
        if not remaining_subparts:
            continue

        block_text, _ = _extract_main_question_block(answer_text, main_no)
        if not block_text or len(_normalize_marker_text(block_text)) < 20:
            continue

        block_result = _map_paper_ii_main_block_subparts(block_text, main_no, remaining_subparts)
        if not isinstance(block_result, dict):
            continue

        normalized_block = _normalize_marker_text(block_text)
        remaining_ids = {item["id"] for item in remaining_subparts}
        for key, value in block_result.items():
            if key not in remaining_ids or not value:
                continue

            value_norm = _normalize_marker_text(value)
            if len(value_norm) < 6 or value_norm not in normalized_block:
                continue

            filtered_result[key] = value
            target = next((item for item in remaining_subparts if item["id"] == key), None)
            logger.info(
                "Recovered Paper II sub-part %s from isolated main-question block %s.",
                target.get("label") if target else key,
                main_no,
            )


def _fill_missing_paper_ii_subparts_from_surfaced_candidates(
    answer_text: str,
    chunk: list[dict],
    filtered_result: dict,
    surfaced_candidates_by_main: dict[str, list[dict]],
    allowed_main_nos: set[str] | None = None,
) -> None:
    """
    New final fallback for Paper II:
    if Gemini already surfaced plausible sub-part answers for a real attempted
    main question, reuse those raw candidates to fill any still-missing sub-part
    slots inside that same main block.
    """
    subparts_by_main: dict[str, list[dict]] = {}
    for item in chunk:
        if not item.get("is_sub_question"):
            continue
        main_no = str(item.get("parent_main_label") or "")
        if not main_no:
            continue
        subparts_by_main.setdefault(main_no, []).append(item)

    for main_no, subparts in subparts_by_main.items():
        if allowed_main_nos is not None and main_no not in allowed_main_nos:
            continue

        remaining_subparts = [item for item in subparts if item["id"] not in filtered_result]
        if not remaining_subparts:
            continue

        raw_candidates = surfaced_candidates_by_main.get(main_no, [])
        if not raw_candidates:
            continue

        block_text, _ = _extract_main_question_block(answer_text, main_no)
        if not block_text:
            continue

        existing_norms = {
            _normalize_marker_text(filtered_result[item["id"]])
            for item in subparts
            if item["id"] in filtered_result and filtered_result.get(item["id"])
        }

        unique_candidates = {}
        normalized_block = _normalize_marker_text(block_text)
        for candidate in raw_candidates:
            value = str(candidate.get("text") or "").strip()
            value_norm = _normalize_marker_text(value)
            if (
                len(value_norm) < 6
                or value_norm in existing_norms
                or value_norm not in normalized_block
            ):
                continue
            unique_candidates.setdefault(value_norm, candidate)

        if not unique_candidates:
            continue

        sorted_candidates = sorted(
            unique_candidates.values(),
            key=lambda candidate: (
                _find_normalized_substring_index(block_text, candidate["text"])
                if _find_normalized_substring_index(block_text, candidate["text"]) >= 0
                else 10**9
            ),
        )

        for sub_item, candidate in zip(remaining_subparts, sorted_candidates):
            filtered_result[sub_item["id"]] = candidate["text"]
            logger.info(
                "Filled missing Paper II sub-part %s from surfaced main-block candidate for main %s.",
                sub_item.get("label"),
                main_no,
            )


def _run_paper_ii_recovery_passes(
    answer_text: str,
    chunk: list[dict],
    filtered_result: dict,
    deferred_main_block_candidates: list[dict],
    evidenced_main_nos: set[str],
    surfaced_candidates_by_main: dict[str, list[dict]],
) -> None:
    """
    Keep the newer Paper II recovery flow together so cleanup stays scoped to the
    evaluation mapping path without touching older learning-mode helpers.
    """
    allowed_main_nos = {main_no for main_no in evidenced_main_nos if main_no}
    if not allowed_main_nos:
        return

    _reassign_paper_ii_main_block_candidates(
        answer_text,
        deferred_main_block_candidates,
        chunk,
        filtered_result,
    )
    _fill_paper_ii_subparts_from_main_blocks(
        answer_text,
        chunk,
        filtered_result,
        allowed_main_nos=allowed_main_nos,
    )
    _fill_missing_paper_ii_subparts_from_surfaced_candidates(
        answer_text,
        chunk,
        filtered_result,
        surfaced_candidates_by_main,
        allowed_main_nos=allowed_main_nos,
    )

def map_student_answers(answer_text: str, question_structure: list) -> dict:
    """
    Maps student answer text to question IDs using Gemini.
    Processes questions in batches for robustness.
    """
    if not answer_text or not answer_text.strip():
        return {}

    try:
        logger.info("Starting student answer mapping by parts.")
        flat_structure = _simplify_structure_for_prompt(question_structure)
        
        # Group by part to maintain context
        parts_map = {}
        for item in flat_structure:
            part = item.get("part") or "Unknown"
            if part not in parts_map:
                parts_map[part] = []
            parts_map[part].append(item)
            
        all_mappings = {}
        batch_size = 100  # Process all questions in a part together to prevent misattribution
        
        for part_name, part_items in parts_map.items():
            logger.info("Mapping part: %s (%d items)", part_name, len(part_items))
            # New part isolation: repeated numbering restarts between Paper I and
            # Paper II, so validate each part against its own OCR region.
            part_answer_text = _get_part_specific_answer_text(answer_text, part_name)
            for i in range(0, len(part_items), batch_size):
                chunk = part_items[i : i + batch_size]
                logger.info("Processing batch %d for %s", (i // batch_size) + 1, part_name)
                chunk_ids = {item["id"] for item in chunk}
                
                def _call_mapping(c):
                    return gemini_generate(
                        ANSWER_MAPPING_PROMPT.format(
                            structure=json.dumps(c, ensure_ascii=False, indent=2),
                            answer_text=part_answer_text[:35000],
                        ),
                        json_mode=True,
                    )

                response_text = _call_mapping(chunk)
                print(f"=== RAW GEMINI RESPONSE FOR {part_name} ===")
                print(response_text)
                print("=============================================")
                batch_result = _safe_json_loads(response_text) if response_text else None

                # Retry on BOTH empty response AND JSON parse failure
                if not isinstance(batch_result, dict):
                    reason = "empty response" if not response_text else "JSON parse failure"
                    logger.warning("Batch mapping for %s batch %d: %s — retrying...",
                                   part_name, (i // batch_size) + 1, reason)
                    response_text = _call_mapping(chunk)
                    batch_result = _safe_json_loads(response_text) if response_text else None

                if isinstance(batch_result, dict):
                    # We accept any key that the model returned, as long as it's in the original structure chunk
                    filtered_result = {}
                    deferred_main_block_candidates = []
                    evidenced_main_nos: set[str] = set()
                    surfaced_candidates_by_main: dict[str, list[dict]] = {}
                    for k, v in batch_result.items():
                        if k in chunk_ids and v:
                            # Find original question text for hallucination check
                            q_item = next((item for item in chunk if item["id"] == k), None)
                            q_text = q_item.get("text", "") if q_item else ""
                            q_label = q_item.get("label", "") if q_item else ""

                            if _is_hallucinated_question_text(v, q_text):
                                logger.warning(f"Discarding hallucinated answer for ID {k} - matches question text.")
                                continue

                            # New recovery signal: if Gemini surfaced a non-hallucinated
                            # Paper II sub-question at all, treat its parent main number as
                            # worth a block-level retry even if strict marker validation
                            # fails on this first pass.
                            if q_item and q_item.get("is_sub_question") and _is_paper_ii_part(part_name):
                                parent_main = str(q_item.get("parent_main_label") or "")
                                evidenced_main_nos.add(parent_main)
                                surfaced_candidates_by_main.setdefault(parent_main, []).append({
                                    "id": k,
                                    "text": v,
                                    "original_label": q_label,
                                })

                            # New decision split: keep direct marker matches as-is,
                            # but defer Paper II main-block-only hits so they can be
                            # reassigned to sub-parts in their actual OCR order.
                            if _has_direct_marker_supported_mapping(part_answer_text, v, q_label):
                                filtered_result[k] = v
                                if q_item and q_item.get("is_sub_question"):
                                    evidenced_main_nos.add(str(q_item.get("parent_main_label") or ""))
                                continue

                            if (
                                q_item
                                and q_item.get("is_sub_question")
                                and _has_main_question_block_support(part_answer_text, v, q_label)
                            ):
                                evidenced_main_nos.add(str(q_item.get("parent_main_label") or ""))
                                deferred_main_block_candidates.append({
                                    "id": k,
                                    "text": v,
                                    "main_no": str(q_item.get("parent_main_label") or ""),
                                    "original_label": q_label,
                                })
                                continue

                            # Existing guard: if neither direct marker support nor
                            # the deferred Paper II main-block path applies, reject it.
                            if not _has_marker_supported_mapping(part_answer_text, v, q_label):
                                logger.warning(
                                    "Discarding unsupported mapping for ID %s - no matching source marker found for label %s.",
                                    k,
                                    q_label,
                                )
                                continue

                            filtered_result[k] = v

                    if _is_paper_ii_part(part_name):
                        _run_paper_ii_recovery_passes(
                            part_answer_text,
                            chunk,
                            filtered_result,
                            deferred_main_block_candidates,
                            evidenced_main_nos,
                            surfaced_candidates_by_main,
                        )

                    all_mappings.update(filtered_result)
                    initial_supported_count = sum(1 for result_id in batch_result.keys() if result_id in chunk_ids)
                    recovered_count = max(0, len(filtered_result) - initial_supported_count)
                    discarded_count = max(0, initial_supported_count - len(filtered_result) + recovered_count)
                    logger.info(
                        "Batch %d for %s: got %d/%d valid mappings (discarded %d initial outputs, recovered %d via Paper II post-pass).",
                        (i // batch_size) + 1,
                        part_name,
                        len(filtered_result),
                        len(chunk),
                        discarded_count,
                        recovered_count,
                    )
                else:
                    logger.error("Batch %d for %s: FAILED after retry — skipping batch.",
                                 (i // batch_size) + 1, part_name)


        # New post-pass: remove short duplicate snippets that would otherwise
        # be counted in both Paper I and Paper II.
        all_mappings = _suppress_cross_part_duplicate_mappings(all_mappings, flat_structure)
        logger.info("Mapped %d answers total across all parts.", len(all_mappings))
        return all_mappings

    except Exception as e:
        logger.error(f"❌ Error in answer mapping: {e}")
        return {}

def map_student_answers(answer_text: str, question_structure: list) -> dict:
    """
    Maps student answer text to question IDs using a single Gemini request.
    Local validation rejects invalid IDs, empty answers, and question-text echoes.
    Marker support is logged but no longer used as a hard rejection gate.
    """
    if not answer_text or not answer_text.strip():
        return {}

    try:
        logger.info("Starting student answer mapping with chunked Gemini requests.")
        flat_structure = _simplify_structure_for_prompt(question_structure)
        by_id = {item["id"]: item for item in flat_structure}
        exact_label_lookup, loose_label_lookup = _build_unique_label_lookup(flat_structure)
        part_text_cache: dict[str, str] = {}
        filtered_result = {}
        max_requests = max(1, int(getattr(settings, "EVAL_GEMINI_ANSWER_MAPPING_MAX_REQUESTS", 1) or 1))
        request_count = min(max_requests, max(1, len(flat_structure)))
        chunk_size = max(1, (len(flat_structure) + request_count - 1) // request_count)
        rejected_counts = {
            "invalid_id": 0,
            "empty_answer": 0,
            "question_text": 0,
            "source_unverified": 0,
            "source_verified": 0,
            "label_resolved": 0,
            "json_failed": 0,
            "local_recovered": 0,
        }

        chunks = [flat_structure[i:i + chunk_size] for i in range(0, len(flat_structure), chunk_size)]
        logger.info(
            "Answer mapping will use %d Gemini request(s) for %d requested questions (chunk_size=%d).",
            len(chunks),
            len(flat_structure),
            chunk_size,
        )

        for chunk_index, chunk in enumerate(chunks, start=1):
            response_text = gemini_generate_evaluation(
                ANSWER_MAPPING_PROMPT.format(
                    structure=json.dumps(chunk, ensure_ascii=False, indent=2),
                    answer_text=answer_text[:35000],
                ),
                budget=EvaluationGeminiClient.ANSWER_MAPPING,
                json_mode=True,
                reason=f"answer_mapping_chunk_{chunk_index}_of_{len(chunks)}",
            )
            print(f"=== RAW GEMINI RESPONSE FOR ANSWER MAPPING CHUNK {chunk_index}/{len(chunks)} ===")
            print(response_text)
            print("========================================================")
            batch_result = _safe_json_loads(response_text) if response_text else None

            if not isinstance(batch_result, (dict, list)):
                rejected_counts["json_failed"] += 1
                logger.error(
                    "Answer mapping chunk %d/%d failed: invalid JSON or empty response.",
                    chunk_index,
                    len(chunks),
                )
                continue

            structured_result = _coerce_answer_mapping(batch_result)
            for raw_key, entry in structured_result.items():
                key, key_source = _resolve_mapping_question_id(entry, by_id, exact_label_lookup, loose_label_lookup)
                if key not in by_id:
                    rejected_counts["invalid_id"] += 1
                    logger.warning(
                        "Discarding mapping with invalid question_id=%s label=%s - no requested question ID or unique label match.",
                        raw_key,
                        entry.get("label", ""),
                    )
                    continue
                if key in filtered_result:
                    continue
                if key_source == "label":
                    rejected_counts["label_resolved"] += 1
                    logger.warning(
                        "Recovered mapping by unique label. raw_question_id=%s label=%s resolved_question_id=%s",
                        raw_key,
                        entry.get("label", ""),
                        key,
                    )
                value = _strip_answer_marker(
                    entry.get("answer", ""),
                    entry.get("label", ""),
                    entry.get("source_marker", ""),
                )
                if not value:
                    rejected_counts["empty_answer"] += 1
                    continue
                if _is_invalid_mapped_answer(value):
                    rejected_counts["empty_answer"] += 1
                    logger.warning(
                        "Discarding placeholder/header mapping for ID %s label %s: %s",
                        key,
                        entry.get("label", ""),
                        value[:80],
                    )
                    continue

                q_item = by_id[key]
                q_text = q_item.get("text", "")
                q_label = q_item.get("label", "")
                part_name = q_item.get("part") or "Unknown"
                if part_name not in part_text_cache:
                    part_text_cache[part_name] = _get_part_specific_answer_text(answer_text, part_name)
                part_answer_text = part_text_cache[part_name]

                if _is_hallucinated_question_text(value, q_text):
                    logger.warning("Discarding hallucinated answer for ID %s - matches question text.", key)
                    rejected_counts["question_text"] += 1
                    continue

                source_verified = _has_marker_supported_mapping(part_answer_text, value, q_label)
                if source_verified:
                    rejected_counts["source_verified"] += 1
                else:
                    rejected_counts["source_unverified"] += 1
                    lowered_confidence = min(float(entry.get("confidence") or 0.0), 0.45)
                    logger.warning(
                        "Keeping unverified mapping for ID %s label %s: no tolerant source marker match. source_verified=false confidence=%.2f",
                        key,
                        q_label,
                        lowered_confidence,
                    )

                filtered_result[key] = value

        local_recovered = _map_answers_from_visible_ocr_markers(answer_text, flat_structure, filtered_result)
        for key, value in local_recovered.items():
            q_item = by_id.get(key, {})
            if _is_invalid_mapped_answer(value):
                rejected_counts["empty_answer"] += 1
                continue
            if _is_hallucinated_question_text(value, q_item.get("text", "")):
                rejected_counts["question_text"] += 1
                continue
            filtered_result[key] = value
            rejected_counts["local_recovered"] += 1

        # Visible OCR markers are the most reliable source for straightforward
        # numbered/sub-numbered answers. Overlay them at the end so model label
        # drift cannot copy one sub-answer into another visible sub-part.
        local_visible = _map_answers_from_visible_ocr_markers(
            answer_text,
            flat_structure,
            {},
            overwrite_existing=True,
        )
        for key, value in local_visible.items():
            if _is_invalid_mapped_answer(value):
                continue
            previous = filtered_result.get(key)
            if previous and _normalize_marker_text(previous) != _normalize_marker_text(value):
                logger.info(
                    "Overriding Gemini mapping for ID %s with visible OCR marker answer.",
                    key,
                )
            filtered_result[key] = value

        filtered_result = _suppress_cross_part_duplicate_mappings(filtered_result, flat_structure)
        filtered_result = _cleanup_final_answer_mappings(filtered_result, flat_structure)
        visible_ocr_answers = _count_visible_ocr_answer_blocks(answer_text)
        logger.info(
            "Chunked answer mapping completed. accepted=%d visible_ocr_answers=%d requested_questions=%d requests=%d source_verified=%d source_unverified=%d label_resolved=%d local_recovered=%d rejected_invalid_id=%d rejected_empty=%d rejected_question_text=%d json_failed=%d",
            len(filtered_result),
            visible_ocr_answers,
            len(flat_structure),
            len(chunks),
            rejected_counts["source_verified"],
            rejected_counts["source_unverified"],
            rejected_counts["label_resolved"],
            rejected_counts["local_recovered"],
            rejected_counts["invalid_id"],
            rejected_counts["empty_answer"],
            rejected_counts["question_text"],
            rejected_counts["json_failed"],
        )
        return filtered_result

    except Exception as e:
        logger.error(f"Error in answer mapping: {e}")
        return {}


def _simplify_structure_for_prompt(questions: list) -> list:
    """
    Helper to create a flat list of all questions and sub-questions for the AI prompt.
    """
    flat_list = []
    
    def process_sub_questions(sub_qs, parent_label, parent_part):
        for sq in sub_qs:
            full_label = f"{parent_label}({sq.label})"
            flat_list.append({
                "id": str(sq.id),
                "label": full_label,
                "part": parent_part,  # inherit parent's part for correct grouping
                "text": sq.sub_question_text[:200] if sq.sub_question_text else "",
                # New metadata: keep the parent main label so Paper II sub-parts
                # can be reassigned within the correct numbered answer block.
                "parent_main_label": parent_label,
                "is_sub_question": True,
            })
            # Handle recursive children
            children = getattr(sq, "children", [])
            if children:
                process_sub_questions(children, full_label, parent_part)

    for q in questions:
        q_part = getattr(q, "part_name", "") or ""
        # Add the main question
        flat_list.append({
            "id": str(q.id),
            "label": q.question_number,
            "part": q_part,
            "text": q.question_text[:200] if q.question_text else "",
            # New flag used by duplicate suppression to detect main questions
            # that should contain a fuller answer block.
            "has_sub_questions": bool(getattr(q, "sub_questions", [])),
            "parent_main_label": q.question_number,
            "is_sub_question": False,
        })
        
        # Add all sub-questions recursively, inheriting parent's part
        sub_qs = getattr(q, "sub_questions", [])
        if sub_qs:
            process_sub_questions(sub_qs, q.question_number, q_part)

            
    return flat_list

RUBRIC_EXTRACT_PROMPT = """
You are an expert examiner. Extract the marking scheme (correct answers) from the provided text.
Identify each question number/label and its corresponding "Correct Answer" or "Acceptable Point".

Format: STRICT JSON object where keys are question labels (e.g., "1", "1(a)", "1.ii") and values are the correct answer string.

Guidelines:
- MCQs: Use the option number (1, 2, 3, 4) or text.
- Essays: Concise bullet points of the expected answer.
- Preserve original labels as they appear (e.g., 1අ, 1a).

TEXT:
{content}
"""

def extract_rubric_answers(text: str) -> dict:
    """
    Extracts a mapping of question identifiers to correct answers from rubric text.
    """
    if not text or len(text.strip()) < 10:
        return {}
        
    try:
        logger.info("Extracting structured answers from rubric.")
        response_text = gemini_generate(
            RUBRIC_EXTRACT_PROMPT.format(content=text[:20000]),
            json_mode=True
        )
        if not response_text:
            return {}
            
        return _safe_json_loads(response_text)
    except Exception as e:
        logger.error(f"Error extracting rubric answers: {e}")
        return {}
def _simplify_sub_questions(sub_questions: list) -> list:
    simple_subs = []
    for sq in sub_questions:
        sq_obj = {
            "id": str(sq.id),
            "label": sq.label,
            "text": sq.sub_question_text[:50] if sq.sub_question_text else ""
        }
        children = getattr(sq, "children", [])
        if children:
            sq_obj["children"] = _simplify_sub_questions(children)
        simple_subs.append(sq_obj)
    return simple_subs

def generate_session_title(message_content: str) -> str:
    """
    Generate an intelligent session title using a lightweight Gemini model.
    Uses gemini-1.5-flash-8b for faster processing and reduced rate limiting.
    """
    if not message_content or not message_content.strip():
        return "New Chat"
    
    try:
        # Use lightweight model for title generation to avoid rate limits
        title = gemini_generate_lightweight(message_content[:500])  # Shorter input
        
        # Fallback if generation fails or returns empty
        if not title or len(title) > 80:  # Stricter length limit
            return "New Chat"
            
        logger.info(f"Generated session title: {title}")
        return title
        
    except Exception as e:
        logger.error(f"Error generating session title: {e}")
        return "New Chat"
