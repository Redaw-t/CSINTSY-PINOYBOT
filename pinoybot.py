"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project, which identifies the language of each word in a code-switched Filipino-English text. The function is designed to be called with a list of tokens and returns a list of tags ("ENG", "FIL", "CS", or "OTH").

Model training and feature extraction should be implemented in a separate script. The trained model should be saved and loaded here for prediction.
"""

import os
import pickle
from typing import List
from Features import extract_sentence_features, vectorize

_BUNDLE = None

def _load_bundle():
    global _BUNDLE
    if _BUNDLE is None:
        with open("model.pkl", "rb") as f:
            _BUNDLE = pickle.load(f)
    return _BUNDLE

# Main tagging function
def tag_language(tokens: List[str]) -> List[str]:
    """
    Tags each token in the input list with its predicted language.
    Args:
        tokens: List of word tokens (strings).
    Returns:
        tags: List of predicted tags ("ENG", "FIL", or "OTH"), one per token.
    """
    if not tokens:
        return []

    bundle = _load_bundle()

    # 1. Load your trained model from disk (e.g., using pickle or joblib)
    model      = bundle["model"]
    vectorizer = bundle["vectorizer"]

    # 2. Extract features from the input tokens to create the feature matrix
    feature_dicts = extract_sentence_features(tokens, labels=None)

    # 3. Use the model to predict the tags for each token
    X, _ = vectorize(feature_dicts, vectorizer=vectorizer)
    predicted = model.predict(X)

    # 4. Convert the predictions to a list of strings ("ENG", "FIL", or "OTH")
    #    CS is remapped to OTH as per spec (only ENG, FIL, OTH allowed)
    tags = ["OTH" if tag == "CS" else str(tag) for tag in predicted]

    # 5. Return the list of tags
    return tags

if __name__ == "__main__":
    # Example usage
    sentence = input("Enter a sentence: ")
    example_tokens = sentence.split()
    print("Tokens:", example_tokens)
    tags = tag_language(example_tokens)
    print("Tags  :", tags)