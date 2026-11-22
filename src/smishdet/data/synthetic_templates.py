"""Template catalogue for synthetic messages, modelled on SLCERT/TRCSL scam alerts.

Each key (e.g. "traffic1") has one text per language, so the same message exists in
every language (parallel data) and shares one `template_group` for leakage-free splits.

Slots: {bank} {brand} {telco} {shop} {vehicle} {parcel} {otp} {amount} {deadline}.
`{deadline}` carries its own postposition ("within 24 hours", "පැය 24 ඇතුළත").

WARNING: Sinhala/Tamil/Singlish/Tanglish texts were written without native-speaker
review. Treat them as unreviewed until checked (see docs, annotation guideline).
Brand names are real Sri Lankan institutions, as in the alerts; messages carry <URL>
and <PHONE> placeholders only, never live links.
"""

from __future__ import annotations

from dataclasses import dataclass

from smishdet.domain.schema import Label, ScamType


@dataclass(frozen=True)
class Meta:
    label: Label
    scam_type: ScamType
    amount_kind: str


SM = Label.SMISHING
META: dict[str, Meta] = {
    "traffic1": Meta(SM, ScamType.TRAFFIC_FINE, "fine"),
    "traffic2": Meta(SM, ScamType.TRAFFIC_FINE, "fine"),
    "customs1": Meta(SM, ScamType.CUSTOMS_DELIVERY, "fee"),
    "customs2": Meta(SM, ScamType.CUSTOMS_DELIVERY, "fee"),
    "bank1": Meta(SM, ScamType.BANK_CARD, "tx"),
    "bank2": Meta(SM, ScamType.BANK_CARD, "tx"),
    "prize1": Meta(SM, ScamType.PRIZE_LOTTERY, "prize"),
    "prize2": Meta(SM, ScamType.PRIZE_LOTTERY, "prize"),
    "govt1": Meta(SM, ScamType.GOVT_AID, "aid"),
    "govt2": Meta(SM, ScamType.GOVT_AID, "aid"),
    "job1": Meta(SM, ScamType.JOB_LOAN, "salary"),
    "job2": Meta(SM, ScamType.JOB_LOAN, "loan"),
    "otp1": Meta(SM, ScamType.OTP_ID_HARVEST, "tx"),
    "otp2": Meta(SM, ScamType.OTP_ID_HARVEST, "tx"),
    "spam1": Meta(Label.SPAM, ScamType.NONE, "tx"),
    "spam2": Meta(Label.SPAM, ScamType.NONE, "tx"),
    "spam3": Meta(Label.SPAM, ScamType.NONE, "tx"),
    "ham1": Meta(Label.HAM, ScamType.NONE, "tx"),
    "ham2": Meta(Label.HAM, ScamType.NONE, "tx"),
    "ham3": Meta(Label.HAM, ScamType.NONE, "tx"),
    "ham4": Meta(Label.HAM, ScamType.NONE, "tx"),
    "ham5": Meta(Label.HAM, ScamType.NONE, "tx"),
    "ham6": Meta(Label.HAM, ScamType.NONE, "bill"),
}

# Shared slot pools. Pools that differ per language (deadline) must have equal length
# so a variant keeps the same underlying choices across languages.
POOLS: dict[str, list[str]] = {
    "bank": ["Commercial Bank", "Sampath Bank", "BOC", "People's Bank", "HNB", "NDB", "DFCC Bank"],
    "brand": ["Dialog", "Mobitel", "SLT", "Lanka Lottery", "Daraz"],
    "telco": ["Dialog", "Mobitel", "SLT-Mobitel", "Hutch"],
    "shop": ["Daraz", "Keells", "Cargills", "Singer", "Softlogic", "Abans"],
}
AMOUNTS: dict[str, list[str]] = {
    "fine": ["1,500", "2,500", "3,000", "5,000"],
    "fee": ["450", "650", "950", "1,250", "2,100"],
    "tx": ["4,500", "12,500", "25,000", "38,750", "75,000"],
    "prize": ["100,000", "250,000", "500,000", "1,000,000", "5,000,000"],
    "aid": ["10,000", "20,000", "50,000"],
    "salary": ["150,000", "200,000", "300,000", "450,000"],
    "loan": ["100,000", "500,000", "1,000,000"],
    "bill": ["1,850", "3,200", "6,400", "9,750"],
}
DEADLINES: dict[str, list[str]] = {
    "en": ["within 24 hours", "within 12 hours", "within 2 hours", "today"],
    "si": ["පැය 24 ඇතුළත", "පැය 12 ඇතුළත", "පැය 2 ඇතුළත", "අද දවස ඇතුළත"],
    "ta": ["24 மணி நேரத்திற்குள்", "12 மணி நேரத்திற்குள்", "2 மணி நேரத்திற்குள்", "இன்றைக்குள்"],
    "si-en": ["paya 24 athule", "paya 12 athule", "paya 2 athule", "ada athule"],
    "ta-en": ["24 mani neratthukkul", "12 mani neratthukkul", "2 mani neratthukkul",
              "indru kulla"],
}

TEMPLATES: dict[str, dict[str, str]] = {
    "en": {
        "traffic1": "Sri Lanka Police: A speeding fine of Rs. {amount} has been issued to your vehicle ({vehicle}). Pay {deadline} or your licence will be suspended. <URL>",
        "traffic2": "SL Police e-ticket: A traffic violation was detected by CCTV. Vehicle {vehicle}. Fine Rs. {amount}. Pay now: <URL>",
        "customs1": "Sri Lanka Post: Your parcel ({parcel}) could not be delivered due to an incorrect address. Pay a customs fee of Rs. {amount} and confirm your address: <URL>",
        "customs2": "Customs: A parcel addressed to you is being held. Fee Rs. {amount}. Pay {deadline}. <URL>",
        "bank1": "{bank}: Your credit card has been suspended. To reactivate, log in {deadline}: <URL>",
        "bank2": "{bank} alert: A suspicious transaction (Rs. {amount}) occurred on your account. If this wasn't you, confirm now <URL>",
        "prize1": "Congratulations! You have won Rs. {amount} in the {brand} lucky draw. Claim your prize at <URL>",
        "prize2": "Your number was selected in the {brand} draw! Call <PHONE> {deadline} to claim Rs. {amount}.",
        "govt1": "Government relief: Rs. {amount} assistance is being given to low-income families. Register at <URL>",
        "govt2": "Presidential Fund: Rs. {amount} aid has been approved for you. Register {deadline}: <URL>",
        "job1": "Overseas job opportunities! Monthly salary Rs. {amount}. Apply fast: WhatsApp <PHONE>.",
        "job2": "Loan of Rs. {amount} in 10 minutes, no guarantor needed. Apply now: <URL>",
        "otp1": "{bank}: To verify your account, send your NIC number and OTP to us. Otherwise your account will be closed.",
        "otp2": "Your SIM will be deactivated {deadline}. To reactivate, send your NIC number and the code you receive to <PHONE>.",
        "spam1": "{telco}: 50% off data packages this weekend! Thank you.",
        "spam2": "{shop} special offers! Up to 30% off all items. This weekend only.",
        "spam3": "{telco} caller tune: Set your favourite song for just Rs. 5 a day! Reply YES to activate.",
        "ham1": "{bank}: Your OTP is {otp}. Do not share it with anyone. It expires in 5 minutes.",
        "ham2": "{bank}: Rs. {amount} was debited from your account. If you did not make this transaction, call the bank's official number.",
        "ham3": "Your order from {shop} has been dispatched. It will arrive tomorrow.",
        "ham4": "Amma, I'll be home around 8 tonight. Don't cook.",
        "ham5": "Doctor appointment: your slot is tomorrow at 9.30 am. Please arrive 15 minutes early.",
        "ham6": "Electricity bill: A bill of Rs. {amount} has been issued. Please pay before the due date.",
    },
    "si": {
        "traffic1": "ශ්‍රී ලංකා පොලීසිය: ඔබගේ වාහනය ({vehicle}) සඳහා රු. {amount} ක වේගය ඉක්මවීමේ දඩයක් පනවා ඇත. {deadline} ගෙවන්න, නැතහොත් බලපත්‍රය අත්හිටුවනු ඇත. <URL>",
        "traffic2": "SL Police e-ticket: CCTV හරහා නීති උල්ලංඝනයක් අනාවරණය විය. වාහනය {vehicle}. දඩය රු. {amount}. දැන්ම ගෙවන්න: <URL>",
        "customs1": "ශ්‍රී ලංකා තැපෑල: ඔබගේ පාර්සලය ({parcel}) ලිපිනය වැරදි නිසා බෙදාහැරීමට නොහැකි විය. රු. {amount} ක රේගු ගාස්තුවක් ගෙවා ලිපිනය තහවුරු කරන්න: <URL>",
        "customs2": "රේගුව: ඔබ වෙත පැමිණි පාර්සලය රඳවා තබාගෙන ඇත. ගාස්තුව රු. {amount}. {deadline} ගෙවන්න. <URL>",
        "bank1": "{bank}: ඔබගේ ක්‍රෙඩිට් කාඩ්පත අත්හිටුවා ඇත. නැවත සක්‍රීය කිරීමට {deadline} මෙහි පිවිසෙන්න: <URL>",
        "bank2": "{bank} අවවාදයයි: ඔබගේ ගිණුමෙන් සැක සහිත ගනුදෙනුවක් (රු. {amount}) සිදු විය. ඔබ නොකළේ නම් දැන්ම තහවුරු කරන්න <URL>",
        "prize1": "සුභ පැතුම්! ඔබ {brand} වාසනාවන්ත දිනුම් ඇදීමේදී රු. {amount} ක ත්‍යාගයක් දිනා ඇත. ලබාගැනීමට <URL> වෙත පිවිසෙන්න.",
        "prize2": "ඔබගේ අංකය {brand} දිනුම් ඇදීමෙන් තෝරා ගෙන ඇත! රු. {amount} ලබාගැනීමට {deadline} <PHONE> අමතන්න.",
        "govt1": "රජයේ සහන: අඩු ආදායම්ලාභී පවුල් සඳහා රු. {amount} ක දීමනාවක් ලබා දේ. ලියාපදිංචි වීමට <URL>",
        "govt2": "ජනාධිපති අරමුදල: ඔබට රු. {amount} ක ආධාරයක් අනුමත විය. {deadline} ලියාපදිංචි වන්න: <URL>",
        "job1": "විදේශ රැකියා අවස්ථා! මාසික වැටුප රු. {amount}. ඉක්මනින් ඉල්ලුම් කරන්න: <PHONE> වෙත WhatsApp කරන්න.",
        "job2": "රු. {amount} ක ණයක් විනාඩි 10කින්, ඇපකරුවන් අවශ්‍ය නැත. දැන්ම ඉල්ලුම් කරන්න: <URL>",
        "otp1": "{bank}: ඔබගේ ගිණුම සත්‍යාපනය කිරීමට NIC අංකය සහ OTP අපට එවන්න. නැතහොත් ගිණුම වසා දමනු ඇත.",
        "otp2": "ඔබගේ SIM එක {deadline} අක්‍රිය වේ. නැවත සක්‍රීය කිරීමට NIC අංකය සහ ලැබෙන කේතය <PHONE> වෙත එවන්න.",
        "spam1": "{telco}: මෙම සති අන්තයේ දත්ත ඇසුරුම් මත 50% ක වට්ටමක්! ස්තූතියි.",
        "spam2": "{shop} විශේෂ දීමනා! සියලුම භාණ්ඩ මත 30% දක්වා වට්ටම්. සති අන්තය සඳහා පමණි.",
        "spam3": "{telco} Caller Tune: ඔබේ ප්‍රියතම ගීතය දිනකට රු. 5කට පමණක් සකසන්න! සක්‍රීය කිරීමට YES යවන්න.",
        "ham1": "{bank}: ඔබගේ OTP අංකය {otp}. මෙය කිසිවෙකුට නොදෙන්න. විනාඩි 5කින් කල් ඉකුත් වේ.",
        "ham2": "{bank}: ඔබගේ ගිණුමෙන් රු. {amount} ක් ඩෙබිට් විය. මෙම ගනුදෙනුව ඔබ නොකළේ නම් බැංකුවේ නිල අංකයට අමතන්න.",
        "ham3": "{shop} වෙතින් ඔබගේ ඇණවුම යවා ඇත. හෙට ලැබෙනු ඇත.",
        "ham4": "අම්මේ, මම අද රෑ 8ට විතර ගෙදර එනවා. කෑම හදන්න එපා.",
        "ham5": "වෛද්‍ය හමුවීම: ඔබගේ වේලාව හෙට උදේ 9.30ට. කරුණාකර විනාඩි 15කට කලින් පැමිණෙන්න.",
        "ham6": "විදුලි බිල: රු. {amount} ක බිලක් නිකුත් කර ඇත. නියමිත දිනට පෙර ගෙවන්න.",
    },
    "ta": {
        "traffic1": "இலங்கை பொலிஸ்: உங்கள் வாகனத்திற்கு ({vehicle}) வேகம் மீறியதற்காக ரூ. {amount} அபராதம் விதிக்கப்பட்டுள்ளது. {deadline} செலுத்தவும், இல்லையெனில் உரிமம் இடைநிறுத்தப்படும். <URL>",
        "traffic2": "SL Police e-ticket: CCTV மூலம் போக்குவரத்து விதிமீறல் கண்டறியப்பட்டது. வாகனம் {vehicle}. அபராதம் ரூ. {amount}. இப்போதே செலுத்தவும்: <URL>",
        "customs1": "இலங்கை தபால்: உங்கள் பார்சல் ({parcel}) முகவரி தவறானதால் வழங்க முடியவில்லை. ரூ. {amount} சுங்கக் கட்டணம் செலுத்தி முகவரியை உறுதிப்படுத்தவும்: <URL>",
        "customs2": "சுங்கத் துறை: உங்களுக்கு வந்த பார்சல் தடுத்து வைக்கப்பட்டுள்ளது. கட்டணம் ரூ. {amount}. {deadline} செலுத்தவும். <URL>",
        "bank1": "{bank}: உங்கள் கிரெடிட் கார்டு இடைநிறுத்தப்பட்டுள்ளது. மீண்டும் செயல்படுத்த {deadline} இங்கே உள்நுழையவும்: <URL>",
        "bank2": "{bank} எச்சரிக்கை: உங்கள் கணக்கில் சந்தேகத்திற்குரிய பரிவர்த்தனை (ரூ. {amount}) நடந்துள்ளது. நீங்கள் செய்யவில்லை என்றால் உடனே உறுதிப்படுத்தவும் <URL>",
        "prize1": "வாழ்த்துகள்! நீங்கள் {brand} அதிர்ஷ்ட குலுக்கலில் ரூ. {amount} பரிசு வென்றுள்ளீர்கள். பெற <URL> க்குச் செல்லவும்.",
        "prize2": "உங்கள் எண் {brand} குலுக்கலில் தேர்வாகியுள்ளது! ரூ. {amount} பெற {deadline} <PHONE> ஐ அழைக்கவும்.",
        "govt1": "அரசு நிவாரணம்: குறைந்த வருமானக் குடும்பங்களுக்கு ரூ. {amount} உதவித்தொகை வழங்கப்படுகிறது. பதிவு செய்ய <URL>",
        "govt2": "ஜனாதிபதி நிதி: உங்களுக்கு ரூ. {amount} உதவி அங்கீகரிக்கப்பட்டுள்ளது. {deadline} பதிவு செய்யவும்: <URL>",
        "job1": "வெளிநாட்டு வேலை வாய்ப்புகள்! மாத சம்பளம் ரூ. {amount}. உடனே விண்ணப்பிக்கவும்: <PHONE> க்கு WhatsApp செய்யவும்.",
        "job2": "ரூ. {amount} கடன் 10 நிமிடங்களில், உத்தரவாதம் தேவையில்லை. இப்போதே விண்ணப்பிக்கவும்: <URL>",
        "otp1": "{bank}: உங்கள் கணக்கை சரிபார்க்க NIC எண்ணையும் OTP யையும் எங்களுக்கு அனுப்புங்கள். இல்லையெனில் கணக்கு மூடப்படும்.",
        "otp2": "உங்கள் SIM {deadline} செயலிழக்கும். மீண்டும் செயல்படுத்த NIC எண்ணையும் வரும் குறியீட்டையும் <PHONE> க்கு அனுப்புங்கள்.",
        "spam1": "{telco}: இந்த வார இறுதியில் டேட்டா பேக்குகளுக்கு 50% தள்ளுபடி! நன்றி.",
        "spam2": "{shop} சிறப்புச் சலுகைகள்! அனைத்துப் பொருட்களுக்கும் 30% வரை தள்ளுபடி. இந்த வார இறுதி மட்டும்.",
        "spam3": "{telco} Caller Tune: உங்களுக்குப் பிடித்த பாடலை நாளொன்றுக்கு ரூ. 5 மட்டுமே! செயல்படுத்த YES அனுப்பவும்.",
        "ham1": "{bank}: உங்கள் OTP எண் {otp}. இதை யாருடனும் பகிர வேண்டாம். 5 நிமிடங்களில் காலாவதியாகும்.",
        "ham2": "{bank}: உங்கள் கணக்கிலிருந்து ரூ. {amount} பற்று வைக்கப்பட்டது. இந்தப் பரிவர்த்தனையை நீங்கள் செய்யவில்லை என்றால் வங்கியின் அதிகாரப்பூர்வ எண்ணை அழைக்கவும்.",
        "ham3": "{shop} இலிருந்து உங்கள் ஆர்டர் அனுப்பப்பட்டுள்ளது. நாளை கிடைக்கும்.",
        "ham4": "அம்மா, நான் இன்று இரவு 8 மணிக்கு வீட்டிற்கு வருவேன். சமைக்க வேண்டாம்.",
        "ham5": "மருத்துவ சந்திப்பு: உங்கள் நேரம் நாளை காலை 9.30. 15 நிமிடங்களுக்கு முன் வரவும்.",
        "ham6": "மின் கட்டணம்: ரூ. {amount} பில் வழங்கப்பட்டுள்ளது. இறுதித் தேதிக்கு முன் செலுத்தவும்.",
    },
    "si-en": {
        "traffic1": "Sri Lanka Police: oyage vehicle eka ({vehicle}) speed limit eka ikmawwata Rs. {amount} dandayak thiyenawa. {deadline} gewanna, nattam licence eka suspend wenawa. <URL>",
        "traffic2": "SL Police e-ticket: CCTV walin traffic rule eka kadala. Vehicle {vehicle}. Dandaya Rs. {amount}. Dan gewanna: <URL>",
        "customs1": "Sri Lanka Post: oyage parcel eka ({parcel}) address eka waradi nisa deliver karanna ba. Rs. {amount} customs gaana gewala address eka confirm karanna: <URL>",
        "customs2": "Customs: oyata awa parcel eka hold karala thiyenawa. Gaana Rs. {amount}. {deadline} gewanna. <URL>",
        "bank1": "{bank}: oyage credit card eka suspend karala. Aye activate karanna {deadline} methana login wenna: <URL>",
        "bank2": "{bank} alert: oyage account eken sandehasahitha transaction ekak (Rs. {amount}) una. Oya nokale nam dan confirm karanna <URL>",
        "prize1": "Congrats! Oya {brand} lucky draw eke Rs. {amount} prize ekak jayagena thiyenawa. Ganna <URL> ekata yanna.",
        "prize2": "Oyage number eka {brand} draw ekata select una! Rs. {amount} ganna {deadline} <PHONE> ekata call karanna.",
        "govt1": "Government relief: adu adayam labana pawul walata Rs. {amount} dimanawak denawa. Register wenna <URL>",
        "govt2": "Presidential fund: oyata Rs. {amount} udawwak approve una. {deadline} register wenna: <URL>",
        "job1": "Videsha job opportunities! Masika salary Rs. {amount}. Ikmanata apply karanna: <PHONE> ekata WhatsApp karanna.",
        "job2": "Loan Rs. {amount} minute 10kin, guarantor one na. Dan apply karanna: <URL>",
        "otp1": "{bank}: oyage account eka verify karanna NIC number ekai OTP ekai apita yawanna. Nattam account eka close wenawa.",
        "otp2": "Oyage SIM eka {deadline} cancel wenawa. Aye activate karanna NIC number ekai ena code ekai <PHONE> ekata yawanna.",
        "spam1": "{telco}: me weekend eke data packages walata 50% discount ekak! Thanks.",
        "spam2": "{shop} special offers! Hama item ekakatama 30% wethath discount. Weekend eka sandaha witharai.",
        "spam3": "{telco} Caller Tune: oyage priyatama gee eka dawasakata Rs. 5kata witharak set karanna! Activate karanna YES yawanna.",
        "ham1": "{bank}: oyage OTP eka {otp}. Eka kawuruwath ekka share karanna epa. Minute 5kin expire wenawa.",
        "ham2": "{bank}: oyage account eken Rs. {amount} debit una. Me transaction eka oya nokale nam bank eke official number ekata call karanna.",
        "ham3": "{shop} eken oyage order eka dispatch karala. Heta labenawa.",
        "ham4": "Amma, mama ada raa 8ta wage gedara enawa. Kama hadanna epa.",
        "ham5": "Doctor appointment: oyage time eka heta udee 9.30ta. Karunakara minute 15kata kalin enna.",
        "ham6": "Light bill: Rs. {amount} bill ekak issue karala. Due date ekata kalin gewanna.",
    },
    "ta-en": {
        "traffic1": "Sri Lanka Police: ungal vehicle ({vehicle}) speed limit meeri Rs. {amount} fine vithikkapattullathu. {deadline} pay pannunga, illaiyel licence suspend aagum. <URL>",
        "traffic2": "SL Police e-ticket: CCTV moolam traffic rule meeral kandupidikkapattathu. Vehicle {vehicle}. Fine Rs. {amount}. Ippove pay pannunga: <URL>",
        "customs1": "Sri Lanka Post: ungal parcel ({parcel}) address thappa irukkirathal deliver panna mudiyala. Rs. {amount} customs fee pay panni address confirm pannunga: <URL>",
        "customs2": "Customs: ungalukku vandha parcel hold la irukku. Fee Rs. {amount}. {deadline} pay pannunga. <URL>",
        "bank1": "{bank}: ungal credit card suspend aagirukku. Marubadiyum activate panna {deadline} inge login pannunga: <URL>",
        "bank2": "{bank} alert: ungal account la sandhegamana transaction (Rs. {amount}) nadandhirukku. Neenga pannala na ippove confirm pannunga <URL>",
        "prize1": "Congrats! Neenga {brand} lucky draw la Rs. {amount} prize jeyichirukkeenga. Peravum <URL> poonga.",
        "prize2": "Ungal number {brand} draw la select aagirukku! Rs. {amount} peravum {deadline} <PHONE> ku call pannunga.",
        "govt1": "Government relief: kuraindha varumaanam ulla kudumbangalukku Rs. {amount} udhavi tharappadukirathu. Register panna <URL>",
        "govt2": "Presidential fund: ungalukku Rs. {amount} udhavi approve aagirukku. {deadline} register pannunga: <URL>",
        "job1": "Foreign job opportunities! Maadha salary Rs. {amount}. Udane apply pannunga: <PHONE> ku WhatsApp pannunga.",
        "job2": "Loan Rs. {amount} 10 minute la, guarantor thevai illa. Ippove apply pannunga: <URL>",
        "otp1": "{bank}: ungal account verify panna NIC number um OTP um engalukku anuppunga. Illaiyel account close aagum.",
        "otp2": "Ungal SIM {deadline} deactivate aagum. Marubadiyum activate panna NIC number um varum code um <PHONE> ku anuppunga.",
        "spam1": "{telco}: indha weekend data packs ku 50% discount! Nandri.",
        "spam2": "{shop} special offers! Ella items kum 30% varai discount. Indha weekend mattum.",
        "spam3": "{telco} Caller Tune: ungalukku pidicha paatta naalukku Rs. 5 mattume! Activate panna YES anuppunga.",
        "ham1": "{bank}: ungal OTP {otp}. Idhai yaarodum share panna vendam. 5 minute la expire aagum.",
        "ham2": "{bank}: ungal account la irundhu Rs. {amount} debit aagirukku. Indha transaction neenga pannala na bank oda official number ku call pannunga.",
        "ham3": "{shop} la irundhu ungal order dispatch aagiduchu. Naalaikku kidaikkum.",
        "ham4": "Amma, naan indru iravu 8 mani ku veetuku varuven. Samaikka vendam.",
        "ham5": "Doctor appointment: ungal neram naalai kaalai 9.30. 15 minute mun vaanga.",
        "ham6": "Current bill: Rs. {amount} bill issue aagirukku. Kadaisi thethikku mun pay pannunga.",
    },
}
