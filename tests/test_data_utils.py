from smishdet.data.pii import scrub
from smishdet.data.script import detect_script
from smishdet.domain.schema import Script


def test_detect_script_sinhala():
    assert detect_script("ඔබගේ ගිණුම අත්හිටුවා ඇත") == Script.SINHALA


def test_detect_script_tamil():
    assert detect_script("உங்கள் கணக்கு முடக்கப்பட்டது") == Script.TAMIL


def test_detect_script_latin_and_mixed():
    assert detect_script("Your account is blocked") == Script.LATIN
    assert detect_script("Your ගිණුම blocked") == Script.MIXED


def test_scrub_pii():
    out = scrub("Pay at http://bit.ly/x call 0771234567 NIC 199012345678")
    assert "<URL>" in out and "<PHONE>" in out and "<NIC>" in out
    assert "0771234567" not in out
