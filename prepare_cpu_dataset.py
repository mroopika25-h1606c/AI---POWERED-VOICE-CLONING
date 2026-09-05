from pathlib import Path
import pandas as pd

DATA_PROJECT = Path(
    r"C:\Users\Manoj M N\OneDrive\Desktop\voiceguard-data-testing"
)

INPUT_FOLDER = Path("metadata/member5")
OUTPUT_FOLDER = Path("metadata")
OUTPUT_FOLDER.mkdir(exist_ok=True)

SPLITS = {
    "train": {
        "input": "train.csv",
        "output": "LA_cpu_train.csv",
        "real": 500,
        "fake": 500,
        "split": "train",
    },
    "validation": {
        "input": "validation.csv",
        "output": "LA_cpu_val.csv",
        "real": 100,
        "fake": 100,
        "split": "val",
    },
    "test": {
        "input": "test.csv",
        "output": "LA_cpu_test.csv",
        "real": 200,
        "fake": 200,
        "split": "test",
    },
}


def make_audio_path(file_path):
    clean_path = str(file_path).replace("\\", "/")
    full_path = DATA_PROJECT / Path(clean_path)
    return str(full_path.resolve())


def prepare_split(config):
    input_path = INPUT_FOLDER / config["input"]
    output_path = OUTPUT_FOLDER / config["output"]

    if not input_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    if "status" in df.columns:
        df = df[
            df["status"].astype(str).str.lower() == "valid"
        ].copy()

    df["temporary_label"] = (
        df["label"].astype(str).str.lower().str.strip()
    )

    real_df = df[
        df["temporary_label"].isin(["real", "bonafide"])
    ]

    fake_df = df[
        df["temporary_label"].isin(["fake", "spoof"])
    ]

    if len(real_df) < config["real"]:
        raise ValueError("Not enough real audio files")

    if len(fake_df) < config["fake"]:
        raise ValueError("Not enough fake audio files")

    selected_real = real_df.sample(
        n=config["real"],
        random_state=42,
    )

    selected_fake = fake_df.sample(
        n=config["fake"],
        random_state=42,
    )

    selected = pd.concat(
        [selected_real, selected_fake],
        ignore_index=True,
    )

    selected = selected.sample(
        frac=1,
        random_state=42,
    ).reset_index(drop=True)

    result = pd.DataFrame()

    result["file_id"] = selected["audio_id"]

    result["label"] = selected["temporary_label"].map({
        "real": "bonafide",
        "bonafide": "bonafide",
        "fake": "spoof",
        "spoof": "spoof",
    })

    result["label_int"] = result["label"].map({
        "bonafide": 0,
        "spoof": 1,
    })

    result["split"] = config["split"]
    result["dataset_name"] = "LA_cpu"

    result["audio_path"] = selected["file_path"].apply(
        make_audio_path
    )

    result["file_exists"] = result["audio_path"].apply(
        lambda path: Path(path).exists()
    )

    missing_files = result[
        result["file_exists"] == False
    ]

    print(f"\n{config['split'].upper()}")
    print(f"Selected files: {len(result)}")
    print(result["label"].value_counts())
    print(f"Missing audio files: {len(missing_files)}")

    if len(missing_files) > 0:
        print("\nFirst missing audio path:")
        print(missing_files.iloc[0]["audio_path"])

        raise FileNotFoundError(
            "Audio paths are incorrect. Dataset creation stopped."
        )

    result = result.drop(columns=["file_exists"])
    result.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")


for split_information in SPLITS.values():
    prepare_split(split_information)

print("\nCPU DATASET CREATED SUCCESSFULLY")