from tools.filename_validator import validate_filenames
from tools.json_validator import validate_json
from tools.language_validator import validate_language_code


def test_filename_validator_reports_missing_and_unexpected() -> None:
    assert validate_filenames({"a.mp3"}, {"b.mp3"}) == {"missing": ["a.mp3"], "unexpected": ["b.mp3"]}


def test_json_and_language_validators() -> None:
    assert validate_json('{"title": "Story"}', {"title"}) == []
    assert validate_json("not json")
    assert validate_language_code("en")
    assert not validate_language_code("English")
