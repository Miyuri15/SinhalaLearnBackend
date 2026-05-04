import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.components.document_processing.services import classifier_service


def _load_grading_service_class():
    fake_embeddings = types.ModuleType("app.shared.ai.embeddings")
    fake_embeddings.xlmr = MagicMock()
    fake_embeddings.ml_semaphore = None
    fake_embeddings.ensure_sentences_cached = lambda *args, **kwargs: None
    fake_embeddings._embedding_cache = {}
    sys.modules["app.shared.ai.embeddings"] = fake_embeddings

    from app.services.evaluation.grading_service import GradingService

    return GradingService


def test_single_pass_answer_mapping_uses_one_gemini_call(monkeypatch):
    calls = []

    def fake_gemini_generate_evaluation(prompt, *, budget, json_mode=False, reason=None):
        calls.append(
            {
                "budget": budget.duty_name,
                "json_mode": json_mode,
                "reason": reason,
                "model": budget.model_name,
            }
        )
        return '{"q-1": "1. student answer"}'

    monkeypatch.setattr(classifier_service, "gemini_generate_evaluation", fake_gemini_generate_evaluation)
    monkeypatch.setattr(classifier_service, "_has_marker_supported_mapping", lambda *args, **kwargs: True)
    monkeypatch.setattr(classifier_service, "_is_hallucinated_question_text", lambda *args, **kwargs: False)
    monkeypatch.setattr(classifier_service, "_get_part_specific_answer_text", lambda text, part: text)
    monkeypatch.setattr(classifier_service, "_suppress_cross_part_duplicate_mappings", lambda mappings, _: mappings)

    question = SimpleNamespace(
        id="q-1",
        question_number="1",
        part_name="Paper_I",
        question_text="Who was the king?",
        sub_questions=[],
    )

    result = classifier_service.map_student_answers("1. student answer", [question])

    assert result == {"q-1": "1. student answer"}
    assert len(calls) == 1
    assert calls[0]["budget"] == "answer_mapping"
    assert calls[0]["reason"] == "answer_mapping_chunk_1_of_1"
    assert calls[0].get("model", classifier_service.EvaluationGeminiClient.ANSWER_MAPPING.model_name) == classifier_service.EvaluationGeminiClient.ANSWER_MAPPING.model_name


def test_single_pass_answer_mapping_keeps_unverified_structured_output(monkeypatch):
    def fake_gemini_generate_evaluation(prompt, *, budget, json_mode=False, reason=None):
        return (
            '{"mappings": ['
            '{"question_id": "q-2", "label": "2", "answer": "ජේම්ස් හර්ග්‍රීව්ස්.", '
            '"source_marker": "2.", "confidence": 0.9}'
            ']}'
        )

    monkeypatch.setattr(classifier_service, "gemini_generate_evaluation", fake_gemini_generate_evaluation)
    monkeypatch.setattr(classifier_service, "_has_marker_supported_mapping", lambda *args, **kwargs: False)
    monkeypatch.setattr(classifier_service, "_is_hallucinated_question_text", lambda *args, **kwargs: False)
    monkeypatch.setattr(classifier_service, "_get_part_specific_answer_text", lambda text, part: text)
    monkeypatch.setattr(classifier_service, "_suppress_cross_part_duplicate_mappings", lambda mappings, _: mappings)

    question = SimpleNamespace(
        id="q-2",
        question_number="2",
        part_name="Paper_I",
        question_text="Who invented the spinning jenny?",
        sub_questions=[],
    )

    result = classifier_service.map_student_answers("2. ජේම්ස් හර්ග්‍රීව්ස්.", [question])

    assert result == {"q-2": "ජේම්ස් හර්ග්‍රීව්ස්."}


def test_single_pass_answer_mapping_recovers_stale_id_by_unique_label(monkeypatch):
    def fake_gemini_generate_evaluation(prompt, *, budget, json_mode=False, reason=None):
        return (
            '{"mappings": ['
            '{"question_id": "stale-id", "label": "2", "answer": "ජේම්ස් හර්ග්‍රීව්ස්.", '
            '"source_marker": "2.", "confidence": 0.9}'
            ']}'
        )

    monkeypatch.setattr(classifier_service, "gemini_generate_evaluation", fake_gemini_generate_evaluation)
    monkeypatch.setattr(classifier_service, "_has_marker_supported_mapping", lambda *args, **kwargs: True)
    monkeypatch.setattr(classifier_service, "_is_hallucinated_question_text", lambda *args, **kwargs: False)
    monkeypatch.setattr(classifier_service, "_get_part_specific_answer_text", lambda text, part: text)
    monkeypatch.setattr(classifier_service, "_suppress_cross_part_duplicate_mappings", lambda mappings, _: mappings)

    question = SimpleNamespace(
        id="q-2",
        question_number="2",
        part_name="Paper_I",
        question_text="Who invented the spinning jenny?",
        sub_questions=[],
    )

    result = classifier_service.map_student_answers("2. ජේම්ස් හර්ග්‍රීව්ස්.", [question])

    assert result == {"q-2": "ජේම්ස් හර්ග්‍රීව්ස්."}


def test_label_recovery_prefers_exact_zero_padded_labels():
    flat = [
        {"id": "short-1", "label": "1"},
        {"id": "long-1", "label": "01"},
    ]
    exact_lookup, loose_lookup = classifier_service._build_unique_label_lookup(flat)

    assert classifier_service._resolve_mapping_question_id(
        {"question_id": "stale", "label": "1"},
        {},
        exact_lookup,
        loose_lookup,
    ) == ("short-1", "label")
    assert classifier_service._resolve_mapping_question_id(
        {"question_id": "stale", "label": "01"},
        {},
        exact_lookup,
        loose_lookup,
    ) == ("long-1", "label")


def test_safe_json_loads_recovers_largest_partial_mapping():
    parsed = classifier_service._safe_json_loads(
        'prefix {"mappings": [{"question_id": "q-1", "answer": "1. answer"}'
    )

    assert parsed == {"mappings": [{"question_id": "q-1", "answer": "1. answer"}]}


def test_tolerant_marker_matching_supports_zero_padded_and_sinhala_sub_label():
    source = "01. (අ) ගල් අඟුරු තිබුණා\n02. ජේම්ස් හර්ග්‍රීව්ස්."

    assert classifier_service._has_direct_marker_supported_mapping(
        source,
        "ගල් අඟුරු තිබුණා",
        "1(අ)",
    )
    assert classifier_service._has_direct_marker_supported_mapping(
        source,
        "ජේම්ස් හර්ග්‍රීව්ස්.",
        "2",
    )


def test_local_ocr_marker_fallback_recovers_missing_short_and_sub_answers():
    flat_structure = [
        {"id": "short-1", "label": "1", "is_sub_question": False, "has_sub_questions": False},
        {"id": "short-2", "label": "2", "is_sub_question": False, "has_sub_questions": False},
        {"id": "main-1", "label": "1", "is_sub_question": False, "has_sub_questions": True},
        {"id": "sub-1a", "label": "1(a)", "parent_main_label": "1", "is_sub_question": True},
        {"id": "sub-1b", "label": "1(b)", "parent_main_label": "1", "is_sub_question": True},
    ]
    answer_text = (
        "1. coal and iron. 2. james. "
        "01. (a) industry started in Britain. (b) worker unions formed."
    )

    recovered = classifier_service._map_answers_from_visible_ocr_markers(
        answer_text,
        flat_structure,
        existing={"short-1": "already mapped"},
    )

    assert recovered == {
        "short-2": "james",
        "sub-1a": "industry started in Britain",
        "sub-1b": "worker unions formed",
    }
    assert classifier_service._count_visible_ocr_answer_blocks(answer_text) == 4


def test_local_ocr_marker_fallback_splits_inline_numeric_paper_ii_subparts():
    flat_structure = [
        {"id": "q3", "label": "3", "is_sub_question": False, "has_sub_questions": True},
        {"id": "q3-1", "label": "3(1)", "parent_main_label": "3", "is_sub_question": True},
        {"id": "q3-2", "label": "3(2)", "parent_main_label": "3", "is_sub_question": True},
        {"id": "q3-3", "label": "3(3)", "parent_main_label": "3", "is_sub_question": True},
        {"id": "q4", "label": "4", "is_sub_question": False, "has_sub_questions": True},
        {"id": "q4-1", "label": "4(1)", "parent_main_label": "4", "is_sub_question": True},
        {"id": "q4-2", "label": "4(2)", "parent_main_label": "4", "is_sub_question": True},
        {"id": "q4-3", "label": "4(3)", "parent_main_label": "4", "is_sub_question": True},
    ]
    answer_text = (
        "03. (1) mission answer. (2) buddhist revival answer. "
        "(3) hindu and muslim revival answer. "
        "04. (1) colebrooke answer. (2) plantation answer. (3) middle class answer."
    )

    recovered = classifier_service._map_answers_from_visible_ocr_markers(
        answer_text,
        flat_structure,
        existing={},
        overwrite_existing=True,
    )

    assert recovered["q3-2"] == "buddhist revival answer"
    assert recovered["q3-3"] == "hindu and muslim revival answer"
    assert recovered["q4-3"] == "middle class answer"
    assert "04" not in recovered["q3-3"]


def test_local_ocr_marker_fallback_uses_sub_labels_when_db_order_is_shuffled():
    flat_structure = [
        {"id": "q4", "label": "4", "is_sub_question": False, "has_sub_questions": True},
        {"id": "q4-3", "label": "4(3)", "parent_main_label": "4", "is_sub_question": True},
        {"id": "q4-1", "label": "4(1)", "parent_main_label": "4", "is_sub_question": True},
        {"id": "q4-2", "label": "4(2)", "parent_main_label": "4", "is_sub_question": True},
        {"id": "q5", "label": "5", "is_sub_question": False, "has_sub_questions": True},
        {"id": "q5-2", "label": "5(2)", "parent_main_label": "5", "is_sub_question": True},
        {"id": "q5-3", "label": "5(3)", "parent_main_label": "5", "is_sub_question": True},
        {"id": "q5-1", "label": "5(1)", "parent_main_label": "5", "is_sub_question": True},
    ]
    answer_text = (
        "04. (1) colebrooke answer. (2) plantation answer. (3) middle class answer. "
        "05. (1) maccallum answer. (2) donoughmore answer. (3) soulbury answer."
    )

    recovered = classifier_service._map_answers_from_visible_ocr_markers(
        answer_text,
        flat_structure,
        existing={},
        overwrite_existing=True,
    )

    assert recovered["q4-1"] == "colebrooke answer"
    assert recovered["q4-2"] == "plantation answer"
    assert recovered["q4-3"] == "middle class answer"
    assert recovered["q5-1"] == "maccallum answer"
    assert recovered["q5-2"] == "donoughmore answer"
    assert recovered["q5-3"] == "soulbury answer"


def test_question_parsing_uses_configured_model(monkeypatch):
    calls = []

    def fake_gemini_generate(prompt, *, json_mode=False, model_name=None, max_retries=3):
        calls.append(
            {
                "json_mode": json_mode,
                "model_name": model_name,
                "max_retries": max_retries,
            }
        )
        return '{"Paper_I": {"questions": {}}, "Paper_II": {"questions": {}}}'

    monkeypatch.setattr(classifier_service, "gemini_generate", fake_gemini_generate)
    monkeypatch.setattr(
        classifier_service.settings,
        "EVAL_GEMINI_QUESTION_PARSING_MODEL",
        "gemini-2.5-flash",
    )

    result = classifier_service.extract_complete_exam_data("Sample question paper")

    assert result == {
        "Paper_I": {"config": {"total_questions_available": 0}, "questions": {}},
        "Paper_II": {"config": {"total_questions_available": 0}, "questions": {}},
    }
    assert len(calls) == 1
    assert calls[0]["json_mode"] is True
    assert calls[0]["model_name"] == "gemini-2.5-flash"


def test_reference_extraction_respects_request_budget(monkeypatch):
    calls = []

    def fake_gemini_generate_evaluation(prompt, *, budget, json_mode=False, reason=None):
        calls.append(
            {
                "budget": budget.duty_name,
                "reason": reason,
                "model": budget.model_name,
            }
        )
        key_count = prompt.count("--- Key: ref_")
        payload = {f"ref_{idx + 1}": f"Reference {idx + 1}" for idx in range(key_count)}
        import json

        return json.dumps(payload)

    db = MagicMock()
    GradingService = _load_grading_service_class()
    grading_module = sys.modules["app.services.evaluation.grading_service"]
    monkeypatch.setattr(grading_module, "gemini_generate_evaluation", fake_gemini_generate_evaluation)
    monkeypatch.setattr(grading_module.settings, "EVAL_GEMINI_REFERENCE_SCHEMA_MAX_REQUESTS", 2)
    service = GradingService(db)

    targets = []
    for idx in range(6):
        target = SimpleNamespace(correct_answer=None)
        targets.append(
            {
                "key": f"q-{idx}",
                "target": target,
                "question_text": ("Question text " + str(idx) + " ") * 120,
                "max_marks": 5,
                "display_number": str(idx + 1),
                "question_type": "essay",
            }
        )

    result = service._batch_extract_reference_points(targets, "Syllabus content")

    assert len(calls) == 2
    assert all(call["budget"] == "reference_extraction" for call in calls)
    assert all(
        call.get("model", grading_module.EvaluationGeminiClient.REFERENCE_SCHEMA.model_name)
        == grading_module.EvaluationGeminiClient.REFERENCE_SCHEMA.model_name
        for call in calls
    )
    assert len(result) == len(targets)


def test_reference_extraction_prompt_uses_per_question_evidence(monkeypatch):
    prompts = []

    def fake_gemini_generate_evaluation(prompt, *, budget, json_mode=False, reason=None):
        prompts.append(prompt)
        return '{"ref_1": "Reference 1"}'

    db = MagicMock()
    GradingService = _load_grading_service_class()
    grading_module = sys.modules["app.services.evaluation.grading_service"]
    monkeypatch.setattr(grading_module, "gemini_generate_evaluation", fake_gemini_generate_evaluation)
    service = GradingService(db)
    monkeypatch.setattr(
        service,
        "_build_reference_context_block",
        lambda question_info, syllabus_text: f"Evidence for {question_info['key']}",
    )

    result = service._batch_extract_reference_points(
        [
            {
                "key": "q-1",
                "target": SimpleNamespace(correct_answer=None),
                "question_text": "Question text",
                "max_marks": 2,
                "display_number": "1",
                "question_type": "short",
            }
        ],
        "Long syllabus content",
    )

    assert result == {"q-1": "Reference 1"}
    assert len(prompts) == 1
    assert "Evidence:\nEvidence for q-1" in prompts[0]
    assert "Do NOT use internal knowledge." in prompts[0]
    assert "{syllabus_for_prompt}" not in prompts[0]


def test_reference_extraction_drops_not_covered_and_uses_context_fallback(monkeypatch):
    def fake_gemini_generate_evaluation(prompt, *, budget, json_mode=False, reason=None):
        return '{"ref_1": "මෙම සාක්ෂිය තුළ මැකඩම් මාර්ග තැනීම පිළිබඳ තොරතුරු අඩංගු නොවේ."}'

    db = MagicMock()
    GradingService = _load_grading_service_class()
    grading_module = sys.modules["app.services.evaluation.grading_service"]
    monkeypatch.setattr(grading_module, "gemini_generate_evaluation", fake_gemini_generate_evaluation)
    service = GradingService(db)
    monkeypatch.setattr(
        service,
        "_build_reference_context_block",
        lambda question_info, syllabus_text: "Retrieved syllabus evidence",
    )

    result = service._batch_extract_reference_points(
        [
            {
                "key": "q-1",
                "target": SimpleNamespace(correct_answer=None),
                "question_text": "Question text",
                "max_marks": 4,
                "display_number": "1(a)",
                "question_type": "essay",
            }
        ],
        "Long syllabus content",
    )

    assert result == {"q-1": "Retrieved syllabus evidence"}


def test_reference_extraction_drops_additional_sinhala_placeholder_variants(monkeypatch):
    def fake_gemini_generate_evaluation(prompt, *, budget, json_mode=False, reason=None):
        return '{"ref_1": "මෙම සාධක වලින් එකක්වත් ලබා දී ඇති සාක්ෂිවල අඩංගු නොවේ."}'

    db = MagicMock()
    GradingService = _load_grading_service_class()
    grading_module = sys.modules["app.services.evaluation.grading_service"]
    monkeypatch.setattr(grading_module, "gemini_generate_evaluation", fake_gemini_generate_evaluation)
    service = GradingService(db)
    monkeypatch.setattr(
        service,
        "_build_reference_context_block",
        lambda question_info, syllabus_text: "Recovered evidence block",
    )

    result = service._batch_extract_reference_points(
        [
            {
                "key": "q-1",
                "target": SimpleNamespace(correct_answer=None),
                "question_text": "Question text",
                "max_marks": 4,
                "display_number": "1(a)",
                "question_type": "essay",
            }
        ],
        "Long syllabus content",
    )

    assert result == {"q-1": "Recovered evidence block"}


def test_reference_extraction_maps_display_number_keys_back_to_question_ids(monkeypatch):
    def fake_gemini_generate_evaluation(prompt, *, budget, json_mode=False, reason=None):
        return '{"1(අ)": "Reference from display key"}'

    db = MagicMock()
    GradingService = _load_grading_service_class()
    grading_module = sys.modules["app.services.evaluation.grading_service"]
    monkeypatch.setattr(grading_module, "gemini_generate_evaluation", fake_gemini_generate_evaluation)
    service = GradingService(db)

    result = service._batch_extract_reference_points(
        [
            {
                "key": "q-1",
                "target": SimpleNamespace(correct_answer=None),
                "question_text": "Question text",
                "max_marks": 4,
                "display_number": "1.අ",
                "question_type": "essay",
            }
        ],
        "Long syllabus content",
    )

    assert result == {"q-1": "Reference from display key"}


def test_reference_extraction_rejects_raw_page_fallback_context(monkeypatch):
    def fake_gemini_generate_evaluation(prompt, *, budget, json_mode=False, reason=None):
        return "{}"

    db = MagicMock()
    GradingService = _load_grading_service_class()
    grading_module = sys.modules["app.services.evaluation.grading_service"]
    monkeypatch.setattr(grading_module, "gemini_generate_evaluation", fake_gemini_generate_evaluation)
    service = GradingService(db)
    monkeypatch.setattr(
        service,
        "_build_reference_context_block",
        lambda question_info, syllabus_text: "--- PAGE 1 --- cover text ... [Context Clipped]",
    )

    result = service._batch_extract_reference_points(
        [
            {
                "key": "q-1",
                "target": SimpleNamespace(correct_answer=None),
                "question_text": "Question text",
                "max_marks": 4,
                "display_number": "1",
                "question_type": "essay",
            }
        ],
        "Long syllabus content",
    )

    assert result == {}


def test_generate_initial_marking_scheme_passes_target_to_reference_extraction(monkeypatch):
    db = MagicMock()
    GradingService = _load_grading_service_class()
    service = GradingService(db)
    service.marking_refs = MagicMock()

    captured = {}

    def fake_batch_extract(questions, syllabus_text):
        captured["questions"] = questions
        return {"q-1": "Reference"}

    monkeypatch.setattr(service, "_batch_extract_reference_points", fake_batch_extract)

    target = SimpleNamespace(id="target-1", question_number="1", correct_answer=None)
    service.generate_initial_marking_scheme(
        evaluation_session_id=SimpleNamespace(),
        syllabus_text="Syllabus",
        questions=[{"key": "q-1", "target": target, "text": "Question text"}],
    )

    assert captured["questions"][0]["target"] is target


def test_safe_json_loads_repairs_trailing_comma():
    payload = """
    {
      "a": "one",
      "b": "two",
    }
    """

    result = classifier_service._safe_json_loads(payload)

    assert result == {"a": "one", "b": "two"}


def test_normalize_extracted_exam_result_rebalances_plain_questions():
    raw = {
        "Paper_I": {
            "config": {"total_questions_available": 1},
            "questions": {
                "1": {"type": "structured", "text": "Q1", "marks": 1},
                "2": {"type": "structured", "text": "Q2", "marks": 1},
            },
        },
        "Paper_II": {
            "config": {"total_questions_available": 5},
            "questions": {
                "3": {"type": "structured", "text": "Misplaced Paper I question", "marks": 1},
                "1": {
                    "type": "structured",
                    "text": "Main 1",
                    "marks": 10,
                    "sub_questions": {"a": {"text": "Sub A", "marks": 5}},
                },
                "2": {
                    "type": "structured",
                    "text": "Main 2",
                    "marks": 10,
                    "sub_questions": {"a": {"text": "Sub A", "marks": 5}},
                },
                "4": {
                    "type": "structured",
                    "text": "Main 4",
                    "marks": 10,
                    "sub_questions": {"a": {"text": "Sub A", "marks": 5}},
                },
            },
        },
    }

    normalized = classifier_service._normalize_extracted_exam_result(raw)

    assert normalized["Paper_I"]["config"]["total_questions_available"] == 3
    assert normalized["Paper_II"]["config"]["total_questions_available"] == 3
    assert "3" in normalized["Paper_I"]["questions"]
    assert "3" not in normalized["Paper_II"]["questions"]


def test_segment_exam_text_detects_sinhala_shra_paper_headers():
    text = """
11 ශ්‍රේණිය - ඉතිහාසය

I ශ්‍රකාටස - ශ්‍රකටි පිළිතුරු ප්‍රශ්න
1. කෙටි ප්‍රශ්නයකි. (02 ලකුණු)
2. තවත් කෙටි ප්‍රශ්නයකි. (02 ලකුණු)

II ශ්‍රකාටස - විග්‍රහාත්මක රචනා ප්‍රශ්න
01.
(අ) රචනා ප්‍රශ්නයකි. (04 ලකුණු)
(ආ) තවත් උප ප්‍රශ්නයකි. (04 ලකුණු)
"""

    sections = classifier_service._segment_exam_text_by_paper_headers(text)

    assert set(sections) == {"Paper_I", "Paper_II"}
    assert "කෙටි ප්‍රශ්නයකි" in sections["Paper_I"]
    assert "විග්‍රහාත්මක" in sections["Paper_II"]


def test_backfill_missing_segmented_paper_ii_questions():
    text = """
I ශ්‍රකාටස - ශ්‍රකටි පිළිතුරු ප්‍රශ්න
1. කෙටි ප්‍රශ්නයකි. (02 ලකුණු)

II ශ්‍රකාටස - විග්‍රහාත්මක රචනා ප්‍රශ්න
01.
(අ) පළමු රචනා උප ප්‍රශ්නය. (04 ලකුණු)
(ආ) දෙවන උප ප්‍රශ්නය. (04 ලකුණු)
(ඇ) තුන්වන උප ප්‍රශ්නය. (04 ලකුණු)
02.
(අ) දෙවන රචනා ප්‍රශ්නය. (04 ලකුණු)
03.
(අ) තුන්වන රචනා ප්‍රශ්නය. (04 ලකුණු)
"""
    parsed = {
        "Paper_I": {"config": {}, "questions": {"1": {"text": "කෙටි ප්‍රශ්නයකි"}}},
        "Paper_II": {"config": {}, "questions": {"1": {"text": "only one"}}},
    }

    backfilled = classifier_service._backfill_missing_segmented_questions(parsed, text)

    assert set(backfilled["Paper_II"]["questions"]) == {"1", "2", "3"}
    assert backfilled["Paper_II"]["config"]["total_questions_available"] == 3
    assert set(backfilled["Paper_II"]["questions"]["2"]["sub_questions"]) == {"අ"}


def test_backfill_paper_ii_questions_when_numbers_share_line_with_text():
    text = """
I part
1. Short question.

II part
1. Main question one
(1) Sub part one
(2) Sub part two
2. Main question two
1) Numeric sub part one
2) Numeric sub part two
3. Main question three
4. Main question four
5. Main question five
6. Main question six
7. Main question seven
8. Main question eight
"""
    parsed = {
        "Paper_I": {"config": {}, "questions": {"1": {"text": "Short question."}}},
        "Paper_II": {"config": {}, "questions": {"1": {"text": "only one"}}},
    }

    backfilled = classifier_service._backfill_missing_segmented_questions(parsed, text)

    assert set(backfilled["Paper_II"]["questions"]) == {str(i) for i in range(1, 9)}
    assert backfilled["Paper_II"]["config"]["total_questions_available"] == 8
    assert set(backfilled["Paper_II"]["questions"]["2"]["sub_questions"]) == {"1", "2"}


def test_question_paper_parser_splits_inline_numeric_subparts():
    section = (
        "02. (1) First essay subpart. (04 marks) "
        "(2) Second essay subpart. (04 marks) "
        "(3) Third essay subpart. (04 marks) "
        "03. (1) Another first subpart. (04 marks) "
        "(2) Another second subpart. (04 marks) "
        "04. (1) One subpart only. (04 marks) "
        "05. (1) First of four. (04 marks) "
        "(2) Second of four. (04 marks) "
        "(3) Third of four. (04 marks) "
        "(4) Fourth of four. (04 marks) "
        "06. (9) Ninth subpart. (04 marks) "
        "(10) Tenth subpart. (04 marks)"
    )

    questions = classifier_service._extract_structured_main_questions_from_section(section)

    assert set(questions) == {"2", "3", "4", "5", "6"}
    assert set(questions["2"]["sub_questions"]) == {"1", "2", "3"}
    assert questions["2"]["sub_questions"]["2"]["text"].startswith("Second essay subpart")
    assert set(questions["3"]["sub_questions"]) == {"1", "2"}
    assert set(questions["4"]["sub_questions"]) == {"1"}
    assert set(questions["5"]["sub_questions"]) == {"1", "2", "3", "4"}
    assert set(questions["6"]["sub_questions"]) == {"9", "10"}


def test_segment_exam_text_detects_ocr_rakotasa_headers():
    text = """
1 \u0dbb\u0d9a\u0dcf\u0da7\u0dc3 - short answers
1. Short question.

2 \u0dbb\u0d9a\u0dcf\u0da7\u0dc3 - long answers
01.
(1) Long sub question.
"""

    sections = classifier_service._segment_exam_text_by_paper_headers(text)

    assert set(sections) == {"Paper_I", "Paper_II"}
    assert "Long sub question" in sections["Paper_II"]


def test_segment_exam_text_infers_paper_ii_from_zero_padded_restart():
    text = """
History paper
1. Short question one.
2. Short question two.
3. Short question three.
4. Short question four.
5. Short question five.
6. Short question six.
7. Short question seven.
8. Short question eight.
9. Short question nine.
10. Short question ten.
20. Short question twenty.

Noisy OCR section header that is not readable
01.
(1) Long answer sub question one.
(2) Long answer sub question two.
02.
(1) Second long answer.
"""

    sections = classifier_service._segment_exam_text_by_paper_headers(text)

    assert set(sections) == {"Paper_I", "Paper_II"}
    assert "20. Short question twenty" in sections["Paper_I"]
    assert sections["Paper_II"].startswith("01.")
    assert "02." in sections["Paper_II"]


def test_local_question_extraction_repairs_short_answer_ocr_40_series_markers():
    text = """
1. First short question. (02 ලකුණු)
2. Second short question. (02 ලකුණු)
3. Third short question. (02 ලකුණු)
4. Fourth short question. (02 ලකුණු)
5. Fifth short question. (02 ලකුණු)
6. Sixth short question. (02 ලකුණු)
7. Seventh short question. (02 ලකුණු)
8. Eighth short question. (02 ලකුණු)
9. Ninth short question. (02 ලකුණු)
40. Tenth short question. (02 ලකුණු)
41. Eleventh short question. (02 ලකුණු)
12. Twelfth short question. (02 ලකුණු)
13. Thirteenth short question. (02 ලකුණු)
14. Fourteenth short question. (02 ලකුණු)
15. Fifteenth short question. (02 ලකුණු)
16. Sixteenth short question. (02 ලකුණු)
47. Seventeenth short question. (02 ලකුණු)
48. Eighteenth short question. (02 ලකුණු)
19. Nineteenth short question. (02 ලකුණු)
20. Twentieth short question. (02 ලකුණු)
"""

    questions = classifier_service._extract_structured_main_questions_from_section(text)

    assert set(questions) == {str(i) for i in range(1, 21)}
    assert questions["10"]["text"].startswith("Tenth")
    assert questions["17"]["text"].startswith("Seventeenth")
    assert questions["18"]["text"].startswith("Eighteenth")


def test_local_question_extraction_preserves_real_forty_series_markers():
    text = """
38. Thirty eighth question.
39. Thirty ninth question.
40. Fortieth question.
41. Forty first question.
"""

    questions = classifier_service._extract_structured_main_questions_from_section(text)

    assert set(questions) == {"38", "39", "40", "41"}
    assert questions["40"]["text"].startswith("Fortieth")
