"""
loader.py

PinoyBot: Data Loading Utilities
Dependencies:  Panda, Openpyxl


Windows 11 cmd: 
pip install pandas --break-system-packages
pip install openpyxl

Sample Use:
python loader.py "PinoyBot Team 69 Riel Annotated.xlsx"

This module is responsible for loading the manually annotated dataset
(the "word_id, sentence_id, sentence, word, annotation" spreadsheets) and
turning them into structures that are convenient for feature extraction
and model training, regardless of which classifier the group ends up
using

This file does NOT train anything and does NOT contain the tag_language()
function - see features.py for feature extraction and
separate training script for model fitting/evaluation. This keeps loading,
feature engineering, and training cleanly decoupled so anyone can
swap out the model without touching this file

Expected input format (one row per WORD, not per sentence):
    word_id | sentence_id | sentence | word | annotation

Where "annotation" is one of: FIL, ENG, CS, OTH (case-insensitive on
load, normalized to uppercase). Rows that have not been annotated yet
(blank annotation) are dropped automatically, so it is safe to load a
partially-annotated file while your someone is still working on
their portion
"""

import os
import glob
from typing import List, Dict, Tuple, Optional, Iterable

import pandas as pd

# The four valid tags as defined in the project specifications
VALID_TAGS = {"FIL", "ENG", "CS", "OTH"}

# Default sheet name used in the annotation spreadsheet template
DEFAULT_SHEET_NAME = "Annotated Sentences"

# Expected columns (order matters for positional fallback, but lookup
# is done by name wherever possible so column order in the sheet does
# not matter)
EXPECTED_COLUMNS = ["word_id", "sentence_id", "sentence", "word", "annotation"]



# Loading a single annotated file

def load_annotations(path: str, sheet_name: str = DEFAULT_SHEET_NAME,
                      drop_unannotated: bool = True,
                      verbose: bool = True) -> pd.DataFrame:
    """
    Loads a single annotated .xlsx file into a pandas DataFrame.

    Args:
        path: Path to the .xlsx file (e.g. someones annotated portion,
              or the fully merged team file)
        sheet_name: Name of the worksheet containing the annotations
        drop_unannotated: If True (default), rows whose "annotation" cell
              is blank/None are dropped. Set to False if you want to see
              the full sheet including not-yet-annotated rows (e.g. for
              auditing progress)
        verbose: If True, prints a short summary after loading

    Returns:
        A DataFrame with columns [word_id, sentence_id, sentence, word,
        annotation], sorted by (sentence_id, word_id)

    Raises:
        FileNotFoundError: if the path does not exist
        ValueError: if any required column is missing, or if any
            non-blank annotation value is not one of FIL/ENG/CS/OTH
            after normalization.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Annotation file not found: {path}")

    df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")

    # Normalize column names (strip whitespace, lowercase) so small
    # formatting differences between everyones files dont break loading.
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"'{path}' is missing expected column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df[EXPECTED_COLUMNS].copy()

    # Track provenance  useful once multiple everyones files get merged
    df["source_file"] = os.path.basename(path)

    if drop_unannotated:
        before = len(df)
        df = df[df["annotation"].notna()]
        dropped = before - len(df)
        if verbose and dropped:
            print(f"[loader] Dropped {dropped} unannotated row(s) from {path}")

    # Normalize tag casing/whitespace (e.g. "Oth " -> "OTH") and validate
    df["annotation"] = df["annotation"].astype(str).str.strip().str.upper()

    invalid = df[~df["annotation"].isin(VALID_TAGS) & df["annotation"].notna()]
    # After the astype(str) above, truly-blank cells become the literal
    # string "NAN" only if drop_unannotated=False; guard against that here
    invalid = invalid[invalid["annotation"] != "NAN"]
    if len(invalid) > 0:
        bad_examples = invalid[["word_id", "sentence_id", "word", "annotation"]].head(10)
        raise ValueError(
            f"'{path}' contains {len(invalid)} row(s) with an invalid tag "
            f"(must be one of {sorted(VALID_TAGS)}). Examples:\n{bad_examples}"
        )

    df = df.sort_values(["sentence_id", "word_id"]).reset_index(drop=True)

    if verbose:
        n_sentences = df["sentence_id"].nunique()
        print(f"[loader] Loaded {len(df)} annotated word(s) across "
              f"{n_sentences} sentence(s) from {path}")

    return df

# Loading / merging multiple annotators' files

def load_multiple_annotations(paths: Iterable[str],
                               sheet_name: str = DEFAULT_SHEET_NAME,
                               on_duplicate: str = "error",
                               verbose: bool = True) -> pd.DataFrame:
    """
    Loads and merges several annotated files (e.g. one per group member,
    each covering a different range of sentence_id) into a single
    combined DataFrame.

    Args:
        paths: An iterable of file paths to .xlsx annotation files.
        sheet_name: Worksheet name shared by all files.
        on_duplicate: What to do if the same word_id appears (already
            annotated) in more than one file:
              - "error": raise a ValueError (default, safest - forces the
                group to notice overlapping/duplicate work).
              - "keep_first": keep the annotation from whichever file was
                passed in first, drop the rest.
              - "keep_last": keep the annotation from whichever file was
                passed in last.
        verbose: If True, prints a short summary.

    Returns:
        A single merged DataFrame, sorted by (sentence_id, word_id).
    """
    frames = [load_annotations(p, sheet_name=sheet_name, verbose=verbose)
              for p in paths]
    if not frames:
        raise ValueError("No paths were provided to load_multiple_annotations().")

    combined = pd.concat(frames, ignore_index=True)

    dup_mask = combined.duplicated(subset="word_id", keep=False)
    if dup_mask.any():
        dup_ids = sorted(combined.loc[dup_mask, "word_id"].unique().tolist())
        if on_duplicate == "error":
            raise ValueError(
                f"Found {len(dup_ids)} word_id(s) annotated in more than one "
                f"file (e.g. {dup_ids[:10]}...). Sort this out, or "
                f"pass on_duplicate='keep_first'/'keep_last' "
                f"if you're sure this is expected."
            )
        elif on_duplicate == "keep_first":
            combined = combined.drop_duplicates(subset="word_id", keep="first")
        elif on_duplicate == "keep_last":
            combined = combined.drop_duplicates(subset="word_id", keep="last")
        else:
            raise ValueError(f"Unknown on_duplicate option: {on_duplicate!r}")

    combined = combined.sort_values(["sentence_id", "word_id"]).reset_index(drop=True)

    if verbose:
        n_sentences = combined["sentence_id"].nunique()
        print(f"[loader] Combined {len(paths) if hasattr(paths, '__len__') else '?'} "
              f"file(s) into {len(combined)} annotated word(s) across "
              f"{n_sentences} sentence(s)")

    return combined


def load_annotations_from_folder(folder: str, pattern: str = "*.xlsx",
                                  sheet_name: str = DEFAULT_SHEET_NAME,
                                  on_duplicate: str = "error",
                                  verbose: bool = True) -> pd.DataFrame:
    """
    Convenience wrapper: loads every file matching `pattern` inside
    `folder` (e.g. everyones annotated .xlsx dropped in a shared
    "annotations/" folder) and merges them via load_multiple_annotations()
    """
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    if not paths:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in '{folder}'."
        )
    return load_multiple_annotations(paths, sheet_name=sheet_name,
                                      on_duplicate=on_duplicate, verbose=verbose)



# Reshaping into sentences (needed for context-aware features)


def group_by_sentence(df: pd.DataFrame) -> Dict[int, Dict[str, List]]:
    """
    Groups a flat word-level DataFrame back into per-sentence structures
    This is the shape that features.py expects, since several suggested
    features (previous word, next word, position in sentence) require
    looking at a words neighbors

    Returns:
        A dict keyed by sentence_id, where each value is:
            {"tokens": [word1, word2, ...], "labels": [tag1, tag2, ...]}
        Tokens/labels are ordered by word_id (i.e. their original order
        in the sentence)

    Note: if a sentence_id is only partially annotated,
    that sentences tokens/labels will simply be
    shorter than the original sentence filter those out first with
    complete_sentences_only() below if you need whole sentences only.
    """
    grouped: Dict[int, Dict[str, List]] = {}
    for sentence_id, sub in df.groupby("sentence_id"):
        sub = sub.sort_values("word_id")
        grouped[int(sentence_id)] = {
            "tokens": sub["word"].astype(str).tolist(),
            "labels": sub["annotation"].tolist(),
        }
    return grouped


def complete_sentences_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out sentences that are only partially annotated by comparing
    the annotated word count per sentence_id against the total number of
    words that sentence has in the raw 'sentence' text (split on
    whitespace). Useful as a sanity check before training, since a
    half-annotated sentence gives a misleading "previous word" context

    This is a heuristic (whitespace tokenization wont always match the
    original tokenization exactly, e.g. for punctuation), so treat it as
    a warning tool, not a strict guarantee
    """
    def _expected_len(sentence: str) -> int:
        return len(str(sentence).split())

    counts = df.groupby("sentence_id").agg(
        annotated=("word", "count"),
        sentence=("sentence", "first"),
    )
    counts["expected"] = counts["sentence"].apply(_expected_len)
    complete_ids = counts[counts["annotated"] >= counts["expected"]].index
    return df[df["sentence_id"].isin(complete_ids)].reset_index(drop=True)



# Train / validation / test splitting


def train_val_test_split(df: pd.DataFrame,
                          train_size: float = 0.70,
                          val_size: float = 0.15,
                          test_size: float = 0.15,
                          random_state: int = 42
                          ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into train/validation/test sets using the
    70-15-15 ratio required by the project specifications.

    Important: the split is done at the SENTENCE level (by sentence_id),
    not the individual word level. If we split by word instead, words
    from the same sentence could end up in both train and test, which
    leaks context information (e.g. surrounding-word features) between
    the sets and inflates evaluation performance. Grouping by sentence
    avoids that.

    Args:
        df: The combined, annotated DataFrame (see load_annotations /
            load_multiple_annotations).
        train_size, val_size, test_size: Proportions of SENTENCES
            assigned to each split. Must sum to 1.0 (small floating
            point slack is tolerated).
        random_state: Seed for reproducibility  keep this fixed so
            your reported evaluation numbers are reproducible.

    Returns:
        (train_df, val_df, test_df) - three DataFrames with the same
        columns as the input, ready to be passed to group_by_sentence()
        or straight into features.py.
    """
    if abs((train_size + val_size + test_size) - 1.0) > 1e-6:
        raise ValueError("train_size + val_size + test_size must sum to 1.0")

    sentence_ids = df["sentence_id"].unique()
    rng = pd.Series(sentence_ids).sample(frac=1.0, random_state=random_state)
    shuffled_ids = rng.tolist()

    n = len(shuffled_ids)
    n_train = round(n * train_size)
    n_val = round(n * val_size)
    # Give any leftover (rounding) sentences to test, so all sentences
    # are always accounted for.
    train_ids = set(shuffled_ids[:n_train])
    val_ids = set(shuffled_ids[n_train:n_train + n_val])
    test_ids = set(shuffled_ids[n_train + n_val:])

    train_df = df[df["sentence_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["sentence_id"].isin(val_ids)].reset_index(drop=True)
    test_df = df[df["sentence_id"].isin(test_ids)].reset_index(drop=True)

    return train_df, val_df, test_df



# Flat convenience accessors (for Anyone who just

def to_tokens_and_labels(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Flattens a word-level DataFrame into a plain (tokens, labels) pair,
    ignoring sentence boundaries. Handy for a quick baseline model that
    doesnt use any context features. Prefer group_by_sentence() if your
    features use previous/next word information.
    """
    return df["word"].astype(str).tolist(), df["annotation"].tolist()


def label_distribution(df: pd.DataFrame) -> pd.Series:
    """Returns the count of each tag in the dataset - useful for
    spotting class imbalance (e.g. FIL/ENG dominating OTH/CS) before
    training, which affects model and metric choice."""
    return df["annotation"].value_counts()



# Manual test / demo


if __name__ == "__main__":
    import sys

    # Usage: python loader.py path/to/annotated.xlsx [more_files.xlsx ...]
    paths = sys.argv[1:]
    if not paths:
        print("Usage: python loader.py <annotated_file_1.xlsx> [annotated_file_2.xlsx ...]")
        sys.exit(1)

    if len(paths) == 1:
        data = load_annotations(paths[0])
    else:
        data = load_multiple_annotations(paths, on_duplicate="error")

    print("\nTag distribution:")
    print(label_distribution(data))

    train_df, val_df, test_df = train_val_test_split(data)
    print(f"\nSplit sizes (by sentence): "
          f"train={train_df['sentence_id'].nunique()} sentences "
          f"({len(train_df)} words), "
          f"val={val_df['sentence_id'].nunique()} sentences "
          f"({len(val_df)} words), "
          f"test={test_df['sentence_id'].nunique()} sentences "
          f"({len(test_df)} words)")

    sentences = group_by_sentence(data)
    first_sid = sorted(sentences.keys())[0]
    print(f"\nExample - sentence_id {first_sid}:")
    print("tokens:", sentences[first_sid]["tokens"])
    print("labels:", sentences[first_sid]["labels"])