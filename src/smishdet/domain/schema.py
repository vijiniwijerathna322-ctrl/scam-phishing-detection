"""Core domain types. Pure Python: no I/O, no ML dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Label(str, Enum):
    HAM = "ham"
    SPAM = "spam"
    SMISHING = "smishing"


class Language(str, Enum):
    SINHALA = "si"
    TAMIL = "ta"
    ENGLISH = "en"
    SINGLISH = "si-en"  # Sinhala-English code-mixed
    TANGLISH = "ta-en"  # Tamil-English code-mixed
    MIXED = "mixed"


class Script(str, Enum):
    SINHALA = "sinhala"
    TAMIL = "tamil"
    LATIN = "latin"
    MIXED = "mixed"


class ScamType(str, Enum):
    NONE = "none"
    TRAFFIC_FINE = "traffic_fine"
    CUSTOMS_DELIVERY = "customs_delivery"
    BANK_CARD = "bank_card"
    PRIZE_LOTTERY = "prize_lottery"
    GOVT_AID = "govt_aid"
    JOB_LOAN = "job_loan"
    OTP_ID_HARVEST = "otp_id_harvest"
    OTHER = "other"


@dataclass(frozen=True)
class Message:
    id: str
    text: str
    label: Label
    language: Language
    script: Script
    source: str  # e.g. "synthetic", "translated", "public", "donated"
    scam_type: ScamType = ScamType.NONE
    template_group: str | None = None  # used for leakage-free splitting
