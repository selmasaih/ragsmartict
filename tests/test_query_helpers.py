import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import query


def test_tokenize_basic():
    assert query._tokenize("Hello, World! 123") == ["hello", "world", "123"]
    assert query._tokenize("") == []
    assert query._tokenize(None) == []


def test_make_candidate_id_stable():
    meta = {"filename": "a.pdf", "page_number": 2, "chunk_index": 1}
    assert query._make_candidate_id(meta, "fallback") == "a.pdf|2|1"
    assert query._make_candidate_id({}, "fallback") == "fallback"


def test_merge_candidates_keeps_highest_score():
    a = [{"id": "x", "doc": "d1", "meta": {}, "score": 0.2}]
    b = [{"id": "x", "doc": "d1", "meta": {}, "score": 0.9}]
    merged = query._merge_candidates(a, b)
    assert len(merged) == 1
    assert merged[0]["score"] == 0.9


def test_build_user_message_includes_history_and_context():
    msg = query._build_user_message(
        "CTX", [{"role": "user", "content": "salut"}], "ma question"
    )
    assert "Historique" in msg
    assert "CTX" in msg
    assert "ma question" in msg


def test_extractive_answer_first_sentences():
    chunks = ["Phrase une. Phrase deux. Phrase trois."]
    out = query._extractive_answer(chunks, max_sentences=2)
    assert out == "Phrase une. Phrase deux."
    assert query._extractive_answer([]) == ""


def test_strip_think_blocking():
    assert query._strip_think_blocking("<think>plan</think>Réponse") == "Réponse"
    assert query._strip_think_blocking("Pas de think") == "Pas de think"
    # Unclosed think block: everything from the tag onward is dropped.
    assert query._strip_think_blocking("<think>oops never closed") == ""


def _run_filter(pieces):
    feed, flush = query._make_think_filter()
    out = "".join(feed(p) for p in pieces)
    out += flush()
    return out


def test_think_filter_single_chunk():
    assert _run_filter(["<think>secret</think>visible"]) == "visible"


def test_think_filter_split_across_chunks():
    # The opening/closing tags are split across streamed chunks.
    pieces = ["before <th", "ink>hidden rea", "soning</thi", "nk> after"]
    assert _run_filter(pieces) == "before  after"


def test_think_filter_no_tags():
    assert _run_filter(["hello ", "world"]) == "hello world"


def test_strip_echo_passes_normal_answer():
    ans = "La transformée de Fourier décompose un signal [1]."
    assert query._strip_echo(ans) == ans


def test_strip_echo_removes_leaked_scaffold():
    leaked = (
        "Contexte:\n[1] SOURCE: cours.pdf (Page 2)\nblabla\n"
        "Question: c'est quoi?\nRéponse:\nLa vraie réponse [1]."
    )
    assert query._strip_echo(leaked) == "La vraie réponse [1]."


def test_strip_echo_never_swallows_without_marker():
    # Starts with "Contexte" but no "Réponse:" — must not lose content.
    text = "Contexte historique : ce sujet est important et mérite explication."
    assert "important" in query._strip_echo(text)
