# Annotation guidelines (v0.1)

For native-speaker reviewers of the Sinhala / Tamil / code-mixed scam SMS dataset.
Your answers decide how trustworthy the dataset is. Please be honest: marking a bad
message as bad is the most useful thing you can do.

## What you are reviewing
Each row is one short message (SMS style). Some were machine-translated from English,
others written from templates based on public scam alerts (SLCERT, TRCSL). None are real
people's private messages. `<URL>` and `<PHONE>` are placeholders for removed links and
numbers: leave them as they are.

## Open the file correctly
Open `review_sheet.csv` in Excel or Google Sheets (UTF-8, so Sinhala/Tamil show properly).
Fill only the last five columns. Do not edit `id`, `language`, `label_proposed`, `text`.
Save as CSV (UTF-8) when finished.

## Columns to fill

| Column | What to enter |
|---|---|
| `fluency_1to5` | How natural does the message sound to a native speaker? 5 = natural, like a real SMS; 4 = fine, slightly stiff; 3 = understandable but clearly translated; 2 = awkward or partly wrong; 1 = unreadable or mostly not in the language |
| `meaning_kept_yes_no` | `yes` if it means the same as `english_reference`; `no` if meaning is lost or changed (e.g. the call to action is missing, a number is dropped, the sense is reversed). Leave blank for rows without an English reference |
| `label_check` | `ok` if `label_proposed` is correct; otherwise the right label: `ham`, `spam` or `smishing`; or `unsure` |
| `corrected_text` | Optional. Your corrected version in natural language of the same type (same script/style). Keep `<URL>`/`<PHONE>` and numbers. Leave blank if the text is fine or hopeless |
| `notes` | Optional. Anything useful (slang, wrong dialect, odd wording) |

Judge the message first, then look at `label_proposed`, so the label does not steer you.

## Label definitions
- **ham**: a legitimate message. Includes real-looking bank OTPs, debit alerts, delivery
  notices, bills, appointments and personal chat. Hard cases: a bank OTP that says "do not
  share" is ham; a message asking you to send an OTP is smishing.
- **spam**: unwanted but not trying to steal anything: promotions, discounts, ringtone or
  caller-tune offers, mass marketing.
- **smishing**: tries to trick the reader into giving money, credentials, OTP/NIC, or
  clicking a link. Typical Sri Lankan themes: fake police fines, parcel/customs fees, bank
  card suspension or suspicious transaction, prizes/lotteries, government cash aid, job or
  loan offers, SIM deactivation / OTP and NIC requests.

When both spam and smishing seem possible, choose smishing if the message asks for
personal data, a payment, or a click to "verify/claim/pay".

## Language notes
- `si` Sinhala script; `ta` Tamil script.
- `si-en` Singlish and `ta-en` Tanglish: Sinhala/Tamil written in English letters (mixed
  with English words). There is no official spelling, so judge naturalness and meaning, not
  spelling. If it reads like something a person in Sri Lanka would really type, score it high.
- Brand names (Dialog, Commercial Bank) in English letters are normal. Do not penalise them.

## Worked examples
| Message (English meaning) | fluency | meaning | label_check |
|---|---|---|---|
| "Your account is blocked, verify now: <URL>" in natural Sinhala | 5 | yes | ok (smishing) |
| Same message, but the "verify now" part is missing | 3 | no | ok |
| "Dialog: 50% off data this weekend" | 5 | yes | spam if proposed ham |
| "Your OTP is 482913, do not share it" | 5 | yes | ham |
| Output is mostly English words | 1 | no | ok |

## Time and quality
About 140 rows take 45-60 minutes. If a reviewer is unsure about many rows, stop and tell
the project owner: guessing hurts more than skipping. Reviewer agreement will be checked
on a small overlap; two reviewers per language is ideal.

## Privacy
Do not add real phone numbers, names or NIC numbers in `corrected_text` or `notes`.

---

### සිංහල සටහන (කෙටියෙන්)
- හැම පේළියකම පණිවිඩය කියවලා ස්වාභාවිකද (1-5) කියලා දෙන්න. ඉංග්‍රීසි මුල් පණිවිඩයේ අර්ථය තියෙනවාද (yes/no) කියලා කියන්න.
- `label_check` එකේ proposed label එක නිවැරදි නම් `ok` ලියන්න. නැත්නම් නිවැරදි එක (`ham`, `spam`, `smishing`).
- හොඳට නැති පණිවිඩයක් නිවැරදි කරන්න පුළුවන් නම් `corrected_text` එකේ ලියන්න. `<URL>` සහ `<PHONE>` වෙනස් කරන්න එපා.
- සැක නම් `unsure` දාන්න. අනුමාන කරනවාට වඩා මග හැරීම හොඳයි.
