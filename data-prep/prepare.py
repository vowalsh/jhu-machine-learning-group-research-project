import json
from pathlib import Path

import pandas as pd
from remotezip import RemoteZip

# Keep the raw release separate from the tables used in the notebooks.
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "PORTIA_Clean_Release"
OUTPUT = DATA / "processed"

# Use the same PORTIA release and a small, repeatable audio sample.
RECORD = "19204539"
URL = f"https://zenodo.org/api/records/{RECORD}/files/PORTIA.zip/content"
PREFIX = "PORTIA_Clean_Release/"
TASKS = {
    "binary": "Binary_Detection",
    "single": "Single_Vessel_Classification",
    "multilabel": "Multi_Label_Monitoring",
}
SPLITS = ("train", "val", "test")
CLASSES = ("tug", "pilot", "cargo", "passenger", "support", "tanker", "other")
SEED = 42
AUDIO_PER_SPLIT = 30
MAX_AUDIO_BYTES = 500_000_000


def temporal_splits(frame: pd.DataFrame) -> dict[object, str]:
    """Assign recording days to chronological 70/15/15 train, validation, and test sets."""
    days = sorted(frame["window_start"].dt.date.unique())

    # Split by whole days so the same recording cannot appear in multiple sets.
    # Leave at least one day for validation and one for testing.
    train_end = min(max(1, int(len(days) * 0.70)), len(days) - 2)
    val_end = min(train_end + max(1, int(len(days) * 0.15)), len(days) - 1)
    return {
        day: "train" if i < train_end else "val" if i < val_end else "test"
        for i, day in enumerate(days)
    }


def read_manifests() -> pd.DataFrame:
    """Combine the official CSVs, keeping missing and conflicting labels for EDA."""
    manifests = []
    for task, folder in TASKS.items():
        # Keep the original split labels and read vessel IDs as strings.
        parts = [
            pd.read_csv(
                RAW / folder / f"{split}.csv",
                dtype={"primary_mmsi": "string", "all_mmsis": "string"},
            ).assign(task=task, split=split, official_split=split)
            for split in SPLITS
        ]

        task_frame = pd.concat(parts, ignore_index=True)
        task_frame["timestamp"] = pd.to_datetime(task_frame["timestamp"], utc=True)

        # WAV filenames contain the precise start time. Use all recording days
        # to set the temporal boundaries before excluding any labels or sampling.
        task_frame["window_start"] = pd.to_datetime(
            task_frame["audio_path"].str.removesuffix(".wav"),
            format="%Y%m%d_%H%M%S.%f", utc=True,
        )
        task_frame["temporal_split"] = task_frame["window_start"].dt.date.map(
            temporal_splits(task_frame),
        )
        # Flag every occurrence of a WAV whose labels disagree within this task.
        # A missing label and a known label also count as a disagreement.
        targets = {
            "binary": ["binary_label"],
            "single": ["primary_class"],
            "multilabel": list(CLASSES),
        }[task]
        label_counts = task_frame.groupby("audio_path")[targets].transform(
            "nunique", dropna=False,
        )
        task_frame["conflicting_target"] = label_counts.gt(1).any(axis=1)
        manifests.append(task_frame)

    return pd.concat(manifests, ignore_index=True)


def choose_pilot(manifests: pd.DataFrame) -> pd.DataFrame:
    """Sample up to 30 eligible recordings per task and temporal split."""
    # Leave unclear targets out of research, without changing the full EDA table.
    missing_binary = manifests["task"].eq("binary") & manifests["binary_label"].isna()
    eligible = manifests.loc[~(manifests["conflicting_target"] | missing_binary)]
    eligible = eligible.drop_duplicates(["task", "audio_path"])
    groups = eligible.groupby(["task", "temporal_split"], sort=False)

    # Some validation groups have fewer than 30 recordings, so use all of them.
    return pd.concat([
        group.sample(n=min(AUDIO_PER_SPLIT, len(group)), random_state=SEED)
        for _, group in groups
    ], ignore_index=True)


# RemoteZip reads selected files over HTTP ranges instead of downloading the 76 GB ZIP.
with RemoteZip(URL, timeout=60, support_suffix_range=False) as archive:
    metadata = [PREFIX + "Portia_Metadata.json"] + [
        f"{PREFIX}{folder}/{split}.csv" for folder in TASKS.values() for split in SPLITS
    ]
    archive.extractall(DATA, members=metadata)

    # Prepare the full metadata first, then select the recordings for research.
    manifests = read_manifests()
    pilot = choose_pilot(manifests)
    audio = [
        archive.getinfo(PREFIX + "audio/" + name)
        for name in sorted(pilot["audio_path"].unique())
    ]

    # Count shared WAVs only once and check the download size before extracting.
    audio_bytes = sum(info.file_size for info in audio)
    if audio_bytes > MAX_AUDIO_BYTES:
        raise ValueError("Selected audio exceeds the 500 MB project limit")

    # Reuse downloaded files when their sizes match the archive.
    for info in audio:
        path = DATA / info.filename
        if not path.is_file() or path.stat().st_size != info.file_size:
            archive.extract(info, DATA)

    # List all archive WAVs, not just our sample, for the EDA dataset counts.
    inventory = pd.DataFrame([
        {"audio_path": Path(info.filename).name, "bytes": info.file_size}
        for info in archive.infolist()
        if info.filename.startswith(PREFIX + "audio/") and info.filename.endswith(".wav")
    ])

# Export the notebook tables. Keep official_split for comparison, but use the
# chronological split as the pilot's training, validation, and test assignment.
pilot["split"] = pilot["temporal_split"]
OUTPUT.mkdir(parents=True, exist_ok=True)
manifests.to_csv(OUTPUT / "manifests.csv", index=False)
pilot.to_csv(OUTPUT / "pilot.csv", index=False)

# Save the exact sample and settings so the notebooks can check their inputs.
pilot[["audio_path", "task", "official_split"]].rename(
    columns={"official_split": "split"},
).to_csv(DATA / "pilot_selection.csv", index=False)
inventory.to_csv(DATA / "audio_inventory.csv", index=False)
(DATA / "provenance.json").write_text(
    json.dumps({
        "record_id": RECORD,
        "sampling_split": "temporal",
        "seed": SEED,
        "audio_per_task_split": AUDIO_PER_SPLIT,
        "selected_unique_audio": len(audio),
        "selected_audio_bytes": audio_bytes,
    }, indent=2) + "\n",
    encoding="utf-8",
)

# Show the number of rows available for each task and split.
print(f"Prepared {len(manifests):,} manifest rows and {len(pilot)} pilot rows")
print(pilot.groupby(["task", "split"]).size().unstack(fill_value=0))
