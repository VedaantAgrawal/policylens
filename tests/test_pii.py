from policylens.ingest.pii import redact_pii


def test_redacts_ssn():
    text, count = redact_pii("Applicant SSN: 123-45-6789 on file.")
    assert "[REDACTED-SSN]" in text
    assert "123-45-6789" not in text
    assert count == 1


def test_redacts_email():
    text, count = redact_pii("Contact jane.doe@example.com for details.")
    assert "[REDACTED-EMAIL]" in text
    assert "jane.doe@example.com" not in text
    assert count == 1


def test_redacts_phone():
    text, count = redact_pii("Call (555) 123-4567 for assistance.")
    assert "[REDACTED-PHONE]" in text
    assert count == 1


def test_redacts_credit_card():
    text, count = redact_pii("Card number 4111-1111-1111-1111 was charged.")
    assert "[REDACTED-CREDIT_CARD]" in text
    assert count == 1


def test_no_false_positive_on_space_separated_financial_table():
    # Actuarial loss-development tables in 10-Ks are columns of
    # space-separated 3-4 digit numbers — must not look like a phone
    # number or credit card just because the shape happens to line up.
    text, count = redact_pii("718 054 314 497 594 666 4111 1111 1111 1111")
    assert count == 0


def test_redacts_multiple_types_in_one_string():
    text, count = redact_pii("SSN 123-45-6789, email a@b.com, phone 555-123-4567.")
    assert count == 3


def test_no_false_positive_on_model_law_citation():
    text, count = redact_pii("See NAIC Model Law MO-808-2, Section 3.")
    assert count == 0
    assert text == "See NAIC Model Law MO-808-2, Section 3."


def test_no_false_positive_on_date():
    text, count = redact_pii("Effective 2026-08-03, per the bulletin.")
    assert count == 0


def test_no_false_positive_on_dollar_amount():
    text, count = redact_pii("The policy value is $250,000.")
    assert count == 0


def test_clean_text_passes_through_unchanged():
    original = "Insurers must comply with Section 541.001 of the Insurance Code."
    text, count = redact_pii(original)
    assert text == original
    assert count == 0
