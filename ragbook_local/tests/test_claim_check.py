from ragbook.prompting import parse_claim_check_response, strip_unsubstantiated


def test_parse_json_response():
    resp = '["Sentence one.", "Another claim."]'
    out = parse_claim_check_response(resp)
    assert out == ["Sentence one.", "Another claim."]


def test_parse_fallback_lines():
    resp = "UNSUPPORTED: Sentence one.\n- Another claim."
    out = parse_claim_check_response(resp)
    assert out == ["Sentence one.", "Another claim."]


def test_strip_unsubstantiated():
    ans = "This is supported. This is not supported. Final supported." 
    unsupported = ["This is not supported."]
    stripped = strip_unsubstantiated(ans, unsupported)
    assert "This is not supported." not in stripped
    assert "This is supported." in stripped
    assert "Final supported." in stripped
