"""Arabic simple constant definitions."""

# Arabic letters
ALEF: str = "\u0627"
""" Arabic Letter Alef """
BEH: str = "\u0628"
""" Arabic Letter Beh """
TEH: str = "\u062A"
""" Arabic Letter Teh """
THEH: str = "\u062B"
""" Arabic Letter Theh """
JEEM: str = "\u062C"
""" Arabic Letter Jeem """
HAH: str = "\u062D"
""" Arabic Letter Hah """
KHAH: str = "\u062E"
""" Arabic Letter Khah """
DAL: str = "\u062F"
""" Arabic Letter Dal """
THAL: str = "\u0630"
""" Arabic Letter Thal """
REH: str = "\u0631"
""" Arabic Letter Reh """
ZAIN: str = "\u0632"
""" Arabic Letter Zain """
SEEN: str = "\u0633"
""" Arabic Letter Seen """
SHEEN: str = "\u0634"
""" Arabic Letter Sheen """
SAD: str = "\u0635"
""" Arabic Letter Sad """
DAD: str = "\u0636"
""" Arabic Letter Dad """
TAH: str = "\u0637"
""" Arabic Letter Tah """
ZAH: str = "\u0638"
""" Arabic Letter Zah """
AIN: str = "\u0639"
""" Arabic Letter Ain """
GHAIN: str = "\u063A"
""" Arabic Letter Ghain """
FEH: str = "\u0641"
""" Arabic Letter Feh """
QAF: str = "\u0642"
""" Arabic Letter Qaf """
KAF: str = "\u0643"
""" Arabic Letter Kaf """
LAM: str = "\u0644"
""" Arabic Letter Lam """
MEEM: str = "\u0645"
""" Arabic Letter Meem """
NOON: str = "\u0646"
""" Arabic Letter Noon """
HEH: str = "\u0647"
""" Arabic Letter Heh """
WAW: str = "\u0648"
""" Arabic Letter Waw """
YEH: str = "\u064A"
""" Arabic Letter Yeh """
ALEF_MAKSURA: str = "\u0649"
""" Arabic Letter Alef Maksura """
TEH_MARBUTA: str = "\u0629"
""" Arabic Letter Teh Marbuta """
ALEF_MADDA_ABOVE: str = "\u0622"
""" Arabic Letter Alef With Madda Above """
ALEF_HAMZA_ABOVE: str = "\u0623"
""" Arabic Letter Alef With Hamza Above """
ALEF_HAMZA_BELOW: str = "\u0625"
""" Arabic Letter Alef With Hamza Below """
HAMZA: str = "\u0621"
""" Arabic Letter Hamza """
HAMZA_WAW: str = "\u0624"
""" Arabic Letter Waw With Hamza Above """
HAMZA_YA: str = "\u0626"
""" Arabic Letter Yeh With Hamza Above """
TATWEEL: str = "\u0640"
""" Arabic Tatweel """
KASHIDA: str = TATWEEL
""" Alias for Arabic Tatweel """

# More ALEF variations
ALEF_WASLA: str = "\u0671"
""" Arabic Letter Alef Wasla """
ALEF_HAMZA_ABOVE_WAVY: str = "\u0672"
""" Arabic Letter Alef With Wavy Hamza Above """
ALEF_HAMZA_BELOW_WAVY: str = "\u0673"
""" Arabic Letter Alef With Wavy Hamza Below """

# LAM ALEF
LAM_ALEF: str = "\ufefb"
""" Arabic Ligature Lam with Alef Isolated Form """
LAM_ALEF_HAMZA_ABOVE: str = "\ufef7"
""" Arabic Ligature Lam with Alef with Hamza Above Isolated Form """
LAM_ALEF_HAMZA_BELOW: str = "\ufef9"
""" Arabic Ligature Lam with Alef with Hamza Below Isolated Form """
LAM_ALEF_MADDA_ABOVE: str = "\ufef5"
""" Arabic Ligature Lam with Alef with Madda Above Isolated Form """

# Diacritics (Harakat)
FATHATAN: str = "\u064B"
""" Arabic Fathatan """
DAMMATAN: str = "\u064C"
""" Arabic Dammatan """
KASRATAN: str = "\u064D"
""" Arabic Kasratan """
FATHA: str = "\u064E"
""" Arabic Fatha """
DAMMA: str = "\u064F"
""" Arabic Damma """
KASRA: str = "\u0650"
""" Arabic Kasra """
SHADDA: str = "\u0651"
""" Arabic Shadda """
SUKUN: str = "\u0652"
""" Arabic Sukun """
#  + 3 compound shaddat

# Arabic Numbers
ARABIC_ZERO: str = "\u0660"
""" Arabic-Indic Digit Zero """
ARABIC_ONE: str = "\u0661"
""" Arabic-Indic Digit One """
ARABIC_TWO: str = "\u0662"
""" Arabic-Indic Digit Two """
ARABIC_THREE: str = "\u0663"
""" Arabic-Indic Digit Three """
ARABIC_FOUR: str = "\u0664"
""" Arabic-Indic Digit Four """
ARABIC_FIVE: str = "\u0665"
""" Arabic-Indic Digit Five """
ARABIC_SIX: str = "\u0666"
""" Arabic-Indic Digit Six """
ARABIC_SEVEN: str = "\u0667"
""" Arabic-Indic Digit Seven """
ARABIC_EIGHT: str = "\u0668"
""" Arabic-Indic Digit Eight """
ARABIC_NINE: str = "\u0669"
""" Arabic-Indic Digit Nine """

# Arabic Punctuation
ARABIC_COMMA: str = "\u060C"
""" Arabic Comma """
ARABIC_SEMICOLON: str = "\u061B"
""" Arabic Semicolon """
ARABIC_QUESTION_MARK: str = "\u061F"
""" Arabic Question Mark """
TRIPLE_DOT: str = "\u061E"
""" Arabic Triple Dot Punctuation Mark """
ARABIC_DECIMAL_SEPARATOR: str = "\u066B"
""" Arabic Decimal Separator """
ARABIC_THOUSANDS_SEPARATOR: str = "\u066C"
""" Arabic Thousands Separator """
STAR: str = "\u066D"
""" Arabic Five Pointed Star """
ARABIC_FULL_STOP: str = "\u06D4"
""" Arabic Full Stop """
DATE_SEPARATOR: str = "\u060D"
""" Arabic Date Separator """
END_OF_AYAH: str = "\u06DD"
""" Arabic End Of Ayah """
MISRA_SIGN: str = "\u060F"
""" Arabic Sign Misra """
POETIC_VERSE_SIGN: str = "\u060E"
""" Arabic Poetic Verse Sign """
SAJDAH: str = "\u06E9"
""" Arabic Place Of Sajdah """
HIZB_START: str = "\u06DE"
""" Arabic Start Of Rub El Hizb """
ORNATE_LEFT_PARENTHESIS: str = "\uFD3E"
""" Arabic Ornate Left Parenthesis """
ORNATE_RIGHT_PARENTHESIS: str = "\uFD3F"
""" Arabic Ornate Right Parenthesis """
ARABIC_PERCENTAGE: str = "\u066A"
""" Arabic Percent Sign """
ARABIC_DASH: str = "–"
""" Arabic Dash """
ARABIC_END_OF_AYAH: str = "«"
""" Arabic End Of Ayah """
ARABIC_START_OF_AYAH: str = "»"
""" Arabic Start Of Ayah """

# Word Ligatures
LIGATURE_SALLA_KORANIC: str = "\uFDF0"
""" Arabic Ligature Salla Used As Koranic Stop Sign Isolated Form """
LIGATURE_QALA: str = "\uFDF1"
""" Arabic Ligature Qala Used As Koranic Stop Sign Isolated Form """
LIGATURE_ALLAH: str = "\uFDF2"
""" Arabic Ligature Allah Isolated Form """
LIGATURE_AKBAR: str = "\uFDF3"
""" Arabic Ligature Akbar Isolated Form """
LIGATURE_MOHAMMAD: str = "\uFDF4"
""" Arabic Ligature Mohammad Isolated Form """
LIGATURE_SALAM: str = "\uFDF5"
""" Arabic Ligature Salam Isolated Form """
LIGATURE_RASOUL: str = "\uFDF6"
""" Arabic Ligature Rasoul Isolated Form """
LIGATURE_ALAYHE: str = "\uFDF7"
""" Arabic Ligature Alayhe Isolated Form """
LIGATURE_WASALLAM: str = "\uFDF8"
""" Arabic Ligature Wasallam Isolated Form """
LIGATURE_SALLA: str = "\uFDF9"
""" Arabic Ligature Salla Isolated Form """
LIGATURE_SALLALLAHOU: str = "\uFDFA"
""" Arabic Ligature Sallallahou Alayhe Wasallam """
LIGATURE_JALLAJALALOUHOU: str = "\uFDFB"
""" Arabic Ligature Jallajalalouhou """
LIGATURE_RIAL: str = "\uFDFC"
""" Rial Sign """
LIGATURE_BISMILLAH: str = "\uFDFD"
""" Arabic Ligature Bismillah Ar-Rahman Ar-Raheem """

SMALL_LIGATURE_SALLA_KORANIC: str = "\u06D6"
""" Arabic Small High Ligature Sad With Lam With Alef Maksura """
SMALL_LIGATURE_QALA: str = "\u06D7"
""" Arabic Small High Ligature Qaf With Lam With Alef Maksura """
SMALL_WAW: str = "\u06E5"
""" Arabic Small Waw """
SMALL_YEH: str = "\u06E6"
""" Arabic Small Yeh """

# Small Harakat
SMALL_TAH: str = "\u0615"
""" Arabic Small High Tah """
SMALL_LAM_ALEF_YEH: str = "\u0616"
""" Arabic Small High Ligature Alef With Lam With Yeh """
SMALL_ZAIN: str = "\u0617"
""" Arabic Small High Zain """
SMALL_FATHA: str = "\u0618"
""" Arabic Small Fatha """
SMALL_DAMMA: str = "\u0619"
""" Arabic Small Damma """
SMALL_KASRA: str = "\u061A"
""" Arabic Small Kasra """
SMALL_LAM_ALEF_HIGH: str = "\u06D9"
""" Arabic Small High Lam Alef """
SMALL_JEEM_HIGH: str = "\u06DA"
""" Arabic Small High Jeem """
SMALL_THREE_DOTS_HIGH: str = "\u06DB"
""" Arabic Small High Three Dots """
SMALL_MEEM_HIGH_ISOLATED: str = "\u06E2"
""" Arabic Small High Meem Isolated Form """
SMALL_MEEM_HIGH_INITIAL: str = "\u06D8"
""" Arabic Small High Meem Initial Form  """
SMALL_MEEM_LOW: str = "\u06ED"
""" Arabic Small Low Meem """
SMALL_SEEN_LOW: str = "\u06E3"
""" Arabic Small Low Seen """
SMALL_SEEN_HIGH: str = "\u06DC"
""" Arabic Small High Seen """
SMALL_ZERO_ROUNDED_HIGH: str = "\u06DF"
""" Arabic Small High Rounded Zero """
SMALL_ZERO_RECTANGULAR_HIGH: str = "\u06E0"
""" Arabic Small High Upright Rectangular Zero """
SMALL_DOTLESS_HEAD_HIGH: str = "\u06E1"
""" Arabic Small High Dotless Head Of Khah """
SMALL_MADDA: str = "\u06E4"
""" Arabic Small High Madda """
SMALL_YEH_HIGH: str = "\u06E7"
""" Arabic Small High Yeh """
SMALL_NOON: str = "\u06E8"
""" Arabic Small High Noon """
SMALL_V: str = "\u065A"
""" Arabic Vowel Sign Small V Above """
SMALL_V_INVERTED: str = "\u065B"
""" Arabic Vowel Sign Inverted Small V Above """

# More Harakat
SAD_SIGN: str = "\u0610"
""" Arabic Sign Sallallahou Alayhe Wassallam """
AIN_SIGN: str = "\u0611"
""" Arabic Sign Alayhe Assallam """
RAHMATULLAH_SIGN: str = "\u0612"
""" Arabic Sign Rahmatullah Alayhe """
RADI_SIGN: str = "\u0613"
""" Arabic Sign Radi Allahou Anhu """
TAKHALLUS: str = "\u0614"
""" Arabic Sign Takhallus """
MADDAH_ABOVE: str = "\u0653"
""" Arabic Maddah Above """
HAMZA_ABOVE: str = "\u0654"
""" Arabic Hamza Above """
HAMZA_BELOW: str = "\u0655"
""" Arabic Hamza Below """
ALEF_SUBSCRIPT: str = "\u0656"
""" Arabic Subscript Alef """
ALEF_SUPERSCRIPT: str = "\u0670"
""" Arabic Letter Superscript Alef """
DAMMA_INVERTED: str = "\u0657"
""" Arabic Inverted Damma """
NOON_MARK: str = "\u0658"
""" Arabic Mark Noon Ghunna """
ZWARAKAY: str = "\u0659"
""" Arabic Zwarakay """
DOT_BELOW: str = "\u065C"
""" Arabic Vowel Sign Dot Below """
DAMMA_REVERSED: str = "\u065D"
""" Arabic Reversed Damma """
PERCENTAGE_ABOVE: str = "\u065E"
""" Arabic Fatha With Two Dots """
HAMZA_BELOW_WAVY: str = "\u065F"
""" Arabic Wavy Hamza Below """
LOW_STOP: str = "\u06EA"
""" Arabic Empty Centre Low Stop """
HIGH_STOP: str = "\u06EB"
""" Arabic Empty Centre High Stop """
HIGH_STOP_FILLED: str = "\u06EC"
""" Arabic Rounded High Stop With Filled Centre """

# Arabic Dotless letters
DOTLESS_BEH: str = "\u066E"
""" Arabic Letter Dotless Beh """
DOTLESS_TEH: str = DOTLESS_BEH
""" Arabic Letter Dotless Teh """
DOTLESS_THEH: str = DOTLESS_BEH
""" Arabic Letter Dotless Theh """
DOTLESS_JEEM: str = HAH
""" Arabic Letter Dotless Jeem """
DOTLESS_KHAH: str = HAH
""" Arabic Letter Dotless Khah """
DOTLESS_THAL: str = DAL
""" Arabic Letter Dotless Thal """
DOTLESS_ZAIN: str = REH
""" Arabic Letter Dotless Zain """
DOTLESS_SHEEN: str = SEEN
""" Arabic Letter Dotless Sheen """
DOTLESS_DAD: str = SAD
""" Arabic Letter Dotless Dad """
DOTLESS_ZAH: str = TAH
""" Arabic Letter Dotless Zah """
DOTLESS_GHAIN: str = AIN
""" Arabic Letter Dotless Ghain """
DOTLESS_FEH: str = "\u06A1"
""" Arabic Letter Dotless Feh """
DOTLESS_QAF: str = "\u066F"
""" Arabic Letter Dotless Qaf """
DOTLESS_NOON_GHUNNA: str = "\u06BA"
""" Arabic Letter Dotless Noon Ghunna """
DOTLESS_YEH: str = ALEF_MAKSURA
""" Arabic Letter Dotless Yeh """
DOTLESS_TEH_MARBUTA: str = HEH
""" Arabic Letter Dotless TEH_MARBUTA """
