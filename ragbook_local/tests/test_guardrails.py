from ragbook.guardrails import decide_or_ask


def mk_hit(score, doc_id="d1", text="some text"):
    return {"fused_score": score, "payload": {"doc_id": doc_id, "text": text}}


def test_no_hits():
    dec = decide_or_ask("q", [], min_score=0.1)
    assert not dec.should_answer
    assert dec.probing_questions


def test_low_best_score():
    dec = decide_or_ask("q", [mk_hit(0.05)], min_score=0.1)
    assert not dec.should_answer


def test_small_gap_triggers_probe():
    hits = [mk_hit(0.5), mk_hit(0.44)]
    dec = decide_or_ask("q", hits, min_score=0.1, gap_threshold=0.1)
    assert not dec.should_answer
    assert "Ambiguous" in dec.reason


def test_low_overlap_triggers_probe():
    hits = [mk_hit(0.9, text="unrelated content unrelated"), mk_hit(0.2)]
    dec = decide_or_ask("specificterm", hits, min_score=0.1, min_token_overlap=0.2)
    assert not dec.should_answer
    assert "Low lexical overlap" in dec.reason


def test_low_coverage_triggers_probe():
    hits = [mk_hit(0.9, doc_id="A"), mk_hit(0.8, doc_id="A"), mk_hit(0.7, doc_id="A")]
    dec = decide_or_ask("q", hits, min_score=0.1, require_doc_coverage=2)
    assert not dec.should_answer
    assert "Low source coverage" in dec.reason


def test_good_case():
    hits = [mk_hit(0.9, doc_id="A"), mk_hit(0.5, doc_id="B"), mk_hit(0.3, doc_id="C")]
    dec = decide_or_ask("q", hits, min_score=0.1)
    assert dec.should_answer
    assert dec.reason == "OK"
