"""
Usage:
    python trainer.py "PinoyBot Team 69 Annotated.xlsx"
1. Loads the annotated dataset via Loader.py
2. Extracts features via Features.py
3. Trains a Decision Tree classifier
4. Evaluates on validation and test sets
5. Saves the trained model + vectorizer to 'model.pkl'
"""

import sys
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

from Loader import load_annotations, group_by_sentence, train_val_test_split, label_distribution
from Features import build_feature_matrix, vectorize


def train(path: str):
    # 1. Load data
    print("\n=== Loading Data ===")
    df = load_annotations(path)

    print("\nTag distribution (full dataset):")
    print(label_distribution(df))

    # 2. Train / val / test split (by sentence, as required by spec)
    print("\n=== Splitting Data (70 / 15 / 15 by sentence) ===")
    train_df, val_df, test_df = train_val_test_split(df)

    print(f"Train : {train_df['sentence_id'].nunique()} sentences, {len(train_df)} words")
    print(f"Val   : {val_df['sentence_id'].nunique()} sentences, {len(val_df)} words")
    print(f"Test  : {test_df['sentence_id'].nunique()} sentences, {len(test_df)} words")

    # 3. Feature extraction
    print("\n=== Extracting Features ===")
    train_sentences = group_by_sentence(train_df)
    val_sentences   = group_by_sentence(val_df)
    test_sentences  = group_by_sentence(test_df)

    X_train_raw, y_train = build_feature_matrix(train_sentences)
    X_val_raw,   y_val   = build_feature_matrix(val_sentences)
    X_test_raw,  y_test  = build_feature_matrix(test_sentences)

    # Fit the vectorizer ONLY on training data, then transform val/test
    # with the same feature space — avoids data leakage.
    X_train, vectorizer = vectorize(X_train_raw)
    X_val,   _          = vectorize(X_val_raw,  vectorizer=vectorizer)
    X_test,  _          = vectorize(X_test_raw, vectorizer=vectorizer)

    print(f"Feature matrix shape (train): {X_train.shape}")

    # 4. Train Decision Tree
    print("\n=== Training Decision Tree ===")

    # max_depth=20 balances expressiveness vs overfitting for this feature set.
    # min_samples_leaf=2 stops the tree from memorising single-word anomalies.
    # class_weight='balanced' compensates for label imbalance (FIL/ENG dominate CS/OTH).
    model = DecisionTreeClassifier(
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)
    print("Training complete.")

    # 5. Evaluate on validation set (tune hyperparameters here)
    print("\n=== Validation Set Performance ===")
    y_val_pred = model.predict(X_val)
    print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    print(classification_report(y_val, y_val_pred, digits=4))

    # 6. Evaluate on test set (final numbers for the report)
    #    Look at these ONLY ONCE — don't retune based on them.

    print("\n=== Test Set Performance (FINAL) ===")
    y_test_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(classification_report(y_test, y_test_pred, digits=4))

    # 7. Save model + vectorizer together in one file
    save_path = "model.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({"model": model, "vectorizer": vectorizer}, f)
    print(f"\nModel saved to '{save_path}'")
    print("Done! You can now run pinoybot.py.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py <annotated_file.xlsx>")
        sys.exit(1)
    train(sys.argv[1])