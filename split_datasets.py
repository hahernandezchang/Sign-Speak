import argparse
import os
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = (
    os.path.dirname(SCRIPT_DIR)
    if os.path.basename(SCRIPT_DIR).lower() == "hand_tracking"
    else SCRIPT_DIR
)
DATA_DIR = os.path.join(ROOT_DIR, "data")


def is_alphabet_label(label: str) -> bool:
    return len(label) == 1 and label.isalpha()


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    parser = argparse.ArgumentParser(
        description="Split collected ASL CSV into alphabet-only and word-only datasets."
    )
    parser.add_argument("--input-csv", default=os.path.join(DATA_DIR, "asl_data.csv"))
    parser.add_argument("--alphabet-csv", default=os.path.join(DATA_DIR, "asl_alphabet_data.csv"))
    parser.add_argument("--words-csv", default=os.path.join(DATA_DIR, "asl_words_data.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if "label" not in df.columns:
        raise ValueError("CSV must include a label column")

    labels = df["label"].astype(str).str.strip().str.lower()
    alpha_mask = labels.apply(is_alphabet_label)

    df_alpha = df[alpha_mask].copy()
    df_words = df[~alpha_mask].copy()

    df_alpha.to_csv(args.alphabet_csv, index=False)
    df_words.to_csv(args.words_csv, index=False)

    print(f"Total rows    : {len(df)}")
    print(f"Alphabet rows : {len(df_alpha)} -> {args.alphabet_csv}")
    print(f"Word rows     : {len(df_words)} -> {args.words_csv}")


if __name__ == "__main__":
    main()
