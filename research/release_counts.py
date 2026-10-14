import tempfile
from pathlib import Path

import pandas as pd
from remotezip import RemoteZip

# Same release prepare.py downloads. Do not import that module: it fetches audio on import.
RECORD = "19204539"
URL = f"https://zenodo.org/api/records/{RECORD}/files/PORTIA.zip/content"
PREFIX = "PORTIA_Clean_Release/"
TASKS = {
    "binary": "Binary_Detection",
    "single": "Single_Vessel_Classification",
    "multilabel": "Multi_Label_Monitoring",
}
SPLITS = ("train", "val", "test")

# Published page counts. Not used as the result. Printed beside the CSV row counts.
MAY_PAGE = {"windows": 98_297, "binary": 36_778, "single": 39_996, "multilabel": 34_996}
PAPER = {"windows": 260_932, "binary": 36_148, "single": 39_125, "multilabel": 34_996}
V2_PAGE = {"windows": 95_022, "binary": 36_148, "single": 39_125, "multilabel": 34_996}


def csv_members() -> list[str]:
    return [
        f"{PREFIX}{folder}/{split}.csv"
        for folder in TASKS.values()
        for split in SPLITS
    ]


def count_rows(root: Path) -> dict[str, int]:
    counts = {}
    for task, folder in TASKS.items():
        frames = [
            pd.read_csv(root / PREFIX / folder / f"{split}.csv")
            for split in SPLITS
        ]
        counts[task] = int(sum(len(frame) for frame in frames))
    counts["windows"] = int(
        pd.concat(
            [
                pd.read_csv(root / PREFIX / folder / f"{split}.csv", usecols=["audio_path"])
                for folder in TASKS.values()
                for split in SPLITS
            ],
            ignore_index=True,
        )["audio_path"].nunique()
    )
    return counts


def open_archive() -> RemoteZip:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            return RemoteZip(URL, timeout=180, support_suffix_range=False)
        except Exception as exc:
            last_error = exc
            print(f"zip open failed, attempt {attempt}: {type(exc).__name__}")
    raise RuntimeError("could not open the PORTIA zip") from last_error


def main() -> None:
    members = csv_members()
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp)
        with open_archive() as archive:
            archive.extractall(dest, members=members)
        counts = count_rows(dest)

    print(f"record {RECORD}")
    print(f"{'':<12} {'csv rows':>10} {'may page':>10} {'paper':>10} {'v2 page':>10}")
    for key in ("windows", "binary", "single", "multilabel"):
        print(
            f"{key:<12} {counts[key]:>10} {MAY_PAGE[key]:>10} {PAPER[key]:>10} {V2_PAGE[key]:>10}"
        )
    print("windows is unique audio_path across the nine CSVs, not the sum of rows")


if __name__ == "__main__":
    main()
