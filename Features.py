"""
features.py

PinoyBot: Feature Extraction
Dependencies:  SciKit-Learn

Windows 11 cmd: 
pip install scikit-learn

Sample Use
python features.py "annotated_file.xlsx"

This module turns a raw list of word tokens into a feature matrix that a
supervised classifier can train on. It is intentionally MODEL-AGNOSTIC:
it only produces plain Python dicts of features per token, so whichever
model the group ends up choosing (Decision Tree, Naive Bayes, Logistic
Regression, ...) can consume the same output. Turning those dicts into
actual numeric arrays is a one-line job with sklearn DictVectorizer
(see vectorize() near the bottom), which also makes it trivial to save
alongside the trained model so pinoybot.py can reproduce the exact same
feature space at prediction time.

Feature categories implemented, matching the specification suggestions:
    1. Presence of letters       -> letter counts / vowel ratio / etc.
    2. Capitalization            -> first-letter case, ALL CAPS, etc.
    3. Arrangement of letters    -> character n-grams (prefix/suffix)
    4. Surrounding-word context  -> previous/next word properties,
                                     including the previous tag, which
                                     is useful for models that predict
                                     sequentially (see note in
                                     extract_sentence_features()).

Restrictions honored (per specs.pdf, section 2.3.2): nothing in this
file looks a word up in an English/Filipino dictionary or a pretrained
language-ID model to directly decide its language. All features below
are purely structural/statistical properties of the word and its
context, or hand-picked affix patterns (which the specs explicitly
allow - dictionaries are allowed to *inform* feature design, just not
to be used as a lookup table at prediction time).
"""

import re
import string
from typing import List, Dict, Optional, Tuple, Any

VOWELS = set("aeiou")
# A short, hand-picked list of Filipino affixes commonly involved in
# intra-word code-switching (e.g. "nag-march", "pina-explain",
# "napagtripan"). This is a structural pattern list, not a dictionary of
# whole words, and it is only used to build boolean features - it never
# directly decides a tag.
FIL_PREFIXES = [
    "nag", "mag", "napag", "mapag", "pinag", "ipinag", "ipa", "ipina",
    "pina", "naka", "maka", "pang", "pag", "ika", "taga", "ka", "ma",
    "na", "in", "um", "nakaka", "nakiki", "makiki", "nagpa", "magpa",
    "mapa", "napa", "paki", "sang", "tag", "i",
]
FIL_SUFFIXES = [
    "an", "in", "han", "hin", "ng", "nin", "nan",  "nang", "ong",
    "gang",
]
ENG_SUFFIXES = ["ing", "ed", "tion", "sion", "ly", "er", "est", "able", "s"]

# Small closed-class list of English grammatical/function words
# E.g. "We 'might' 'be' cooked"
ENG_FUNCTION_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "done", "will", "would",
    "can", "could", "should", "may", "might", "shall", "must",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "into", "onto", "upon", "about", "above", "below", "under",
    "between", "among", "through", "during", "before", "after",
    "over", "around", "within", "without", "against",
    "and", "or", "but", "so", "yet", "nor", "although", "because",
    "since", "while", "when", "where", "which", "that", "who",
    "not", "no", "never", "neither", "if", "as", "up", "out",
    "this", "that", "these", "those", "it", "its", "i", "we",
    "they", "he", "she", "you", "my", "our", "their", "his", "her",
    "your", "me", "us", "them", "him", "also", "just", "more",
    "than", "then", "very", "too", "here", "there", "now", "still",
    "even", "only", "both",
}
# Filipino grammatical particles, usually always FIL, never ENG
# Same idea as ENG_FUNCTION_WORDS but for Filipino
# E.g. "Pangit ka 'po' 'ba'?"
FIL_PARTICLES = {
    "ang", "ng", "sa", "si", "mga", "ay", "na", "at", "ni", "kay",
    "kaya", "nga", "ba", "raw", "daw", "lang", "naman", "pala",
    "po", "opo", "ho", "yung", "yun", "yon", "yan", "dun", "dito", "diyan",
    "doon","kung", "pero", "para", "dahil", "habang", "kahit", "bagaman",
    "din", "rin",  "na", "pa", "na", "man", "muna", "talaga", "sana", "siguro",
    "parang", "medyo", "sobra",  "grabe", "naman", "lagi", "muli", "uli",

}

# English consonant clusters that don't naturally appear in Filipino
ENG_CLUSTERS = [
    "str", "spr", "ght", "tch", "sch", "chr", "ph", "th", "wh",
    "ck", "sh", "ch", "kn", "wr", "qu", "dge", "tion", "ould",
]

PUNCT_CHARS = set(string.punctuation)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _letter_only(word: str) -> str:
    """Returns just the alphabetic characters of a word, lowercased."""
    return "".join(ch for ch in word if ch.isalpha()).lower()


def _is_punctuation_token(word: str) -> bool:
    """True if the token has no alphanumeric characters at all (e.g. '.', ',', '...', '!?')."""
    return len(word) > 0 and all(ch in PUNCT_CHARS for ch in word)


def _is_emoji_or_symbol(word: str) -> bool:
    """
    Rough heuristic for emoji/symbol tokens: contains characters outside
    the standard ASCII letters/digits/punctuation range. Catches most
    emoji, without needing an external emoji library.
    """
    return any(ord(ch) > 0x2000 for ch in word)


def _word_shape(word: str) -> str:
    """
    Classic NLP 'word shape' feature: maps each character to a category
    so e.g. 'GAB' -> 'XXX', 'Mitra' -> 'Xxxxx', 'na-award' -> 'xx-xxxxx',
    '67' -> 'dd'. Helps the model generalize capitalization/digit
    patterns without memorizing exact words.
    """
    shape_chars = []
    for ch in word:
        if ch.isupper():
            shape_chars.append("X")
        elif ch.islower():
            shape_chars.append("x")
        elif ch.isdigit():
            shape_chars.append("d")
        else:
            shape_chars.append(ch)
    return "".join(shape_chars)


# ---------------------------------------------------------------------------
# Single-word feature extraction
# ---------------------------------------------------------------------------

def word_features(word: str, prefix: str = "") -> Dict[str, Any]:
    """
    Extracts features that only depend on the word itself (no sentence
    context). Used both for the target word and, with a prefix like
    'prev_' or 'next_', to describe its neighbors.

    Args:
        word: The raw token string.
        prefix: Optional string prepended to every feature name, so this
            function can be reused to describe the previous/next word
            without key collisions in the final feature dict.

    Returns:
        A dict of feature_name -> value (numbers, bools, or short
        strings - all of which DictVectorizer can handle).
    """
    word = "" if word is None else str(word)
    letters = _letter_only(word)
    n_chars = len(word)
    n_letters = len(letters)

    feats: Dict[str, Any] = {}

    # --- 1. Presence of letters -------------------------------------------------
    feats["length"] = n_chars
    feats["n_letters"] = n_letters
    n_vowels = sum(1 for ch in letters if ch in VOWELS)
    n_consonants = n_letters - n_vowels
    feats["n_vowels"] = n_vowels
    feats["n_consonants"] = n_consonants
    feats["vowel_ratio"] = _safe_div(n_vowels, n_letters)
    feats["consonant_ratio"] = _safe_div(n_consonants, n_letters)
    # Per-letter counts/ratios (e.g. "number of a" as suggested in the
    # specs). Kept to a-z only; DictVectorizer will happily one-hot/keep
    # only the letters that actually vary across your dataset.
    for ch in string.ascii_lowercase:
        c = letters.count(ch)
        if c:  # omit zero-count letters to keep the dict compact
            feats[f"count_{ch}"] = c
            feats[f"ratio_{ch}"] = _safe_div(c, n_letters)
    # Letters that are rare/absent in native Filipino spelling and are
    # therefore informative for spotting English/loan spellings.
    non_native_letters = set("cfjqvxz")
    feats["n_non_native_letters"] = sum(letters.count(ch) for ch in non_native_letters)
    feats["has_non_native_letter"] = feats["n_non_native_letters"] > 0

    # Individual flags for each non-native letter — more specific than just counting them
    # These letters rarely appear in native Filipino words so each is a strong ENG signal
    feats["has_c"] = "c" in letters  # common in English (cat, clock, etc.)
    feats["has_f"] = "f" in letters  # rare in Filipino (Filipino borrowed words use "p" instead)
    feats["has_v"] = "v" in letters  # very rare in native Filipino words
    feats["has_x"] = "x" in letters  # almost exclusively English
    feats["has_z"] = "z" in letters  # almost exclusively English

    # --- 2. Capitalization --------------------------------------------------
    feats["is_capitalized"] = word[:1].isupper() if word else False
    feats["is_all_upper"] = word.isupper() if n_letters > 1 else False
    feats["is_all_lower"] = word.islower() if n_letters > 0 else False
    n_upper = sum(1 for ch in word if ch.isupper())
    feats["upper_ratio"] = _safe_div(n_upper, n_chars)

    # --- 3. Arrangement of letters (character n-grams / shape) --------------
    feats["prefix_1"] = word[:1].lower()
    feats["prefix_2"] = word[:2].lower()
    feats["prefix_3"] = word[:3].lower()
    feats["suffix_1"] = word[-1:].lower()
    feats["suffix_2"] = word[-2:].lower()
    feats["suffix_3"] = word[-3:].lower()
    feats["word_shape"] = _word_shape(word)
    feats["has_hyphen"] = "-" in word  # e.g. "na-award", "nag-march"
    feats["has_repeated_letter_run"] = bool(re.search(r"(.)\1{2,}", word.lower()))  # "hahaha", "grrr"
    feats["has_reduplication"] = _has_simple_reduplication(letters)

    # Hand-picked Filipino affix pattern flags (structural, not a lookup).
    lower = word.lower()
    feats["starts_with_fil_prefix"] = any(lower.startswith(p) for p in FIL_PREFIXES)
    feats["ends_with_fil_suffix"] = any(lower.endswith(s) for s in FIL_SUFFIXES)
    feats["ends_with_eng_suffix"] = any(lower.endswith(s) for s in ENG_SUFFIXES)
    feats["ends_ing"] = lower.endswith("ing")  # e.g. running, eating, playing
    feats["ends_tion"] = lower.endswith("tion")  # e.g. nation, construction, education
    feats["ends_ed"] = lower.endswith("ed")  # e.g. walked, talked, played
    feats["ends_ly"] = lower.endswith("ly")  # e.g. quickly, slowly, really
    feats["ends_er"] = lower.endswith("er")  # e.g. teacher, player, worker
    feats["ends_est"] = lower.endswith("est")  # e.g. biggest, fastest, tallest

    # Ends with vowel, Filipino words very commonly end in vowels
    # so if a word DOESN'T end in a vowel, it's more likely English
    feats["ends_with_vowel"] = bool(letters) and letters[-1] in VOWELS

    # English consonant clusters that don't naturally occur in Filipino phonology
    feats["has_eng_consonant_cluster"] = any(c in letters for c in ENG_CLUSTERS)

    # Consecutive vowels, English has more of these (ea, ou, ie) than Filipino
    feats["has_consecutive_vowels"] = bool(re.search(r'[aeiou]{2,}', letters))

    # Filipino also code-switches via INFIXES (inserted inside the word),
    # not just prefixes/suffixes - e.g. "pinull" = p + [in] + ull, an
    # -in- infix spliced into the English root "pull" right after the
    # first consonant. Flag words that look like they contain -in-/-um-
    # spliced in near the start (position 1 or 2), which is the usual
    # insertion point for these two infixes.
    feats["has_fil_infix_pattern"] = lower[1:3] in ("in", "um") or lower[2:4] in ("in", "um")

    # --- 4. Token "type" flags (helps separate OTH from the rest) ----------
    feats["is_punctuation"] = _is_punctuation_token(word)
    feats["is_digit"] = word.isdigit()
    feats["has_digit"] = any(ch.isdigit() for ch in word)
    feats["is_hashtag"] = word.startswith("#")
    feats["is_mention"] = word.startswith("@")
    feats["is_emoji_or_symbol"] = _is_emoji_or_symbol(word)
    feats["is_all_caps_word"] = n_letters >= 2 and word.isupper()  # e.g. acronyms, shouted onomatopoeia

    # English function word flag (some of the common conjunctions, prepositions, etc.) these short words are always ENG but look Filipino structurally
    feats["is_eng_function_word"] = lower in ENG_FUNCTION_WORDS

    # Filipino particle flag — these are always FIL, strong signal to avoid misclassifying as ENG
    feats["is_fil_particle"] = lower in FIL_PARTICLES

    if prefix:
        feats = {f"{prefix}{k}": v for k, v in feats.items()}
    return feats


def _has_simple_reduplication(letters: str) -> bool:
    """
    Detects simple whole/partial reduplication patterns common in
    Filipino morphology (e.g. 'sila-sila', 'araw-araw' minus the hyphen
    once letters-only, or partial reduplication like 'tatakbo'). This is
    a lightweight heuristic, not a dictionary lookup: it just checks
    whether the first half of the letter string repeats.
    """
    n = len(letters)
    if n < 4:
        return False
    half = n // 2
    return letters[:half] == letters[half:half * 2]


# ---------------------------------------------------------------------------
# Sentence-level (context-aware) feature extraction
# ---------------------------------------------------------------------------

def extract_word_context_features(tokens: List[str], index: int,
                                   prev_label: Optional[str] = None
                                   ) -> Dict[str, Any]:
    """
    Builds the full feature dict for tokens[index], combining the word
    own features with features describing its neighbors and its position
    in the sentence.

    Args:
        tokens: The full list of tokens for the sentence (raw strings,
            same format tag_language() receives).
        index: Position of the target word within `tokens`.
        prev_label: The tag of the PREVIOUS word, if known. During
            training, pass the previous word gold annotation. During
            sequential prediction (see note below), pass whatever your
            model predicted for the previous word. Pass None for the
            first word of a sentence, or if your model doesn't want to
            use this feature at all.

            Note on sequential prediction: including prev_label makes
            the model context-aware (e.g. "previous word predicted as
            FIL" mentioned in the specs), but it means tag_language()
            must predict tokens one at a time, left to right, feeding
            each prediction back in as prev_label for the next word --
            rather than predicting the whole sentence in one batched
            call. This is entirely optional: if the group model
            doesn't use prev_label, just leave it as None everywhere
            and predict the whole sentence in a single batch as usual.

    Returns:
        A single flat dict of features for tokens[index].
    """
    n = len(tokens)
    word = tokens[index]

    feats: Dict[str, Any] = {}
    feats.update(word_features(word))

    # --- Position in sentence ---
    feats["position"] = index
    feats["rel_position"] = _safe_div(index, max(n - 1, 1))
    feats["is_first_word"] = index == 0
    feats["is_last_word"] = index == n - 1
    feats["sentence_length"] = n

    # --- Previous word ---
    if index > 0:
        prev_word = tokens[index - 1]
        feats.update(word_features(prev_word, prefix="prev_"))
        prev_letters = _letter_only(prev_word)
        feats["prev_ends_in_vowel"] = bool(prev_letters) and prev_letters[-1] in VOWELS
    else:
        feats["prev_is_none"] = True
    feats["prev_label"] = prev_label if prev_label is not None else "NONE"

    # --- Next word ---
    if index < n - 1:
        next_word = tokens[index + 1]
        feats.update(word_features(next_word, prefix="next_"))
        next_letters = _letter_only(next_word)
        feats["next_starts_with_vowel"] = bool(next_letters) and next_letters[0] in VOWELS
    else:
        feats["next_is_none"] = True

    return feats


def extract_sentence_features(tokens: List[str],
                               labels: Optional[List[str]] = None
                               ) -> List[Dict[str, Any]]:
    """
    Extracts features for every token in a sentence at once.

    Args:
        tokens: List of raw word tokens for one sentence.
        labels: Optional list of gold tags, same length as `tokens`.
            When provided (i.e. during training/evaluation, where the
            true tags are known), the previous word TRUE label is used
            for the prev_label context feature - this is standard
            practice ("teacher forcing") and keeps training simple and
            fast (one batched call instead of per-token loops). When
            `labels` is None (i.e. at real prediction time), prev_label
            is left as None for every token, which is the safe default
            for a model trained without the context feature, or for a
            first pass before wiring up sequential prediction (see
            extract_word_context_features docstring if the group wants
            to use predicted-previous-tag as a feature at inference
            time).

    Returns:
        A list of feature dicts, one per token, in the same order as
        `tokens`.
    """
    if labels is not None and len(labels) != len(tokens):
        raise ValueError("tokens and labels must be the same length")

    feature_dicts = []
    for i in range(len(tokens)):
        prev_label = labels[i - 1] if (labels is not None and i > 0) else None
        feature_dicts.append(extract_word_context_features(tokens, i, prev_label=prev_label))
    return feature_dicts


# ---------------------------------------------------------------------------
# Dataset-level convenience wrappers
# ---------------------------------------------------------------------------

def build_feature_matrix(sentences: Dict[int, Dict[str, List]]
                          ) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Convenience wrapper around extract_sentence_features() for a whole
    dataset already grouped by sentence (i.e. the output of
    loader.group_by_sentence()).

    Args:
        sentences: {sentence_id: {"tokens": [...], "labels": [...]}}

    Returns:
        (X, y): X is a flat list of feature dicts (one per word across
        ALL sentences, in sentence order), y is the matching flat list
        of gold labels. Feed X into vectorize() below, or directly into
        any sklearn model that accepts dict-based features (e.g. via a
        DictVectorizer / FeatureHasher / a Pipeline).
    """
    X: List[Dict[str, Any]] = []
    y: List[str] = []
    for sentence_id in sorted(sentences.keys()):
        tokens = sentences[sentence_id]["tokens"]
        labels = sentences[sentence_id]["labels"]
        X.extend(extract_sentence_features(tokens, labels))
        y.extend(labels)
    return X, y


def vectorize(X: List[Dict[str, Any]], vectorizer=None):
    """
    Turns a list of feature dicts into a numeric matrix using sklearn
    DictVectorizer, which one-hot-encodes string features (like
    'prefix_2') and passes numeric features through as-is.

    Args:
        X: List of feature dicts, e.g. from build_feature_matrix().
        vectorizer: An already-fit DictVectorizer (pass this in when
            transforming validation/test data, so it uses the exact
            same feature space learned on the training data). Leave as
            None to fit a new one (only do this on the TRAINING split).

    Returns:
        (matrix, vectorizer): `matrix` is a scipy sparse matrix ready
        for sklearn .fit()/.predict(). `vectorizer` is the fitted
        DictVectorizer - save this together with your trained model
        (e.g. in the same pickle file, or alongside it) so pinoybot.py
        can reproduce the identical feature space at prediction time.

    Example (in your separate training script):
        X_train, y_train = build_feature_matrix(train_sentences)
        X_train_mat, vec = vectorize(X_train)
        model.fit(X_train_mat, y_train)

        X_val, y_val = build_feature_matrix(val_sentences)
        X_val_mat, _ = vectorize(X_val, vectorizer=vec)
        model.score(X_val_mat, y_val)
    """
    from sklearn.feature_extraction import DictVectorizer

    if vectorizer is None:
        vectorizer = DictVectorizer(sparse=True)
        matrix = vectorizer.fit_transform(X)
    else:
        matrix = vectorizer.transform(X)
    return matrix, vectorizer


# ---------------------------------------------------------------------------
# Manual test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from Loader import load_annotations, group_by_sentence

    # Usage: python features.py <annotated_file.xlsx>
    paths = sys.argv[1:]
    if not paths:
        print("Usage: python features.py <annotated_file.xlsx>")
        sys.exit(1)

    df = load_annotations(paths[0])
    sentences = group_by_sentence(df)

    X, y = build_feature_matrix(sentences)
    print(f"\nExtracted features for {len(X)} words across {len(sentences)} sentences")

    first_sid = sorted(sentences.keys())[0]
    tokens = sentences[first_sid]["tokens"]
    labels = sentences[first_sid]["labels"]
    feats = extract_sentence_features(tokens, labels)

    print(f"\nExample - sentence_id {first_sid}:")
    for tok, lab, f in zip(tokens, labels, feats):
        # Print a small, readable subset of each feature dict.
        keys_to_show = ["length", "vowel_ratio", "is_capitalized",
                         "prefix_2", "suffix_2", "has_hyphen",
                         "starts_with_fil_prefix", "is_punctuation", "prev_label"]
        preview = {k: f[k] for k in keys_to_show if k in f}
        print(f"\n{tok!r:15} -> {lab:4} {preview}")

    X_mat, vec = vectorize(X)
    print(f"\nVectorized shape: {X_mat.shape} ({len(vec.get_feature_names_out())} unique feature columns)")