"""Prepare the approved PORTIA development sample; leave test audio unopened."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from remotezip import RemoteZip

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "PORTIA_Clean_Release"
OUTPUT = DATA / "processed"
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
TRAIN_DAYS = [
    "2025-05-16", "2025-05-17", "2025-05-18", "2025-05-19", "2025-05-20",
    "2025-05-21", "2025-05-22", "2025-05-23", "2025-05-27",
]
VAL_DAYS = pd.date_range("2025-05-28", "2025-06-02").strftime("%Y-%m-%d").tolist()
TEST_DAYS = ["2025-06-03", "2025-06-09", "2025-06-10", "2025-06-12"]
DAY_SPLITS = dict.fromkeys(TRAIN_DAYS, "train")
DAY_SPLITS.update(dict.fromkeys(VAL_DAYS, "val"))
DAY_SPLITS.update(dict.fromkeys(TEST_DAYS, "test"))
SEED = 42
TRAIN_ROWS = 300
VAL_ROWS = 200
MAX_AUDIO_BYTES = 3_000_000_000


def read_manifests() -> pd.DataFrame:
    """Keep the official labels and apply one frozen day map to every task."""
    frames = []
    for task, folder in TASKS.items():
        parts = [
            pd.read_csv(RAW / folder / f"{split}.csv", dtype={
                "primary_mmsi": "string", "all_mmsis": "string",
            }).assign(task=task, official_split=split)
            for split in SPLITS
        ]
        frame = pd.concat(parts, ignore_index=True)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        audio_timestamps = frame["audio_path"].str.removesuffix(".wav")
        frame["window_start"] = pd.to_datetime(
            audio_timestamps,
            format="%Y%m%d_%H%M%S.%f", utc=True,
        )
        frame["recording_day"] = frame["window_start"].dt.strftime("%Y-%m-%d")
        frame["time_bin"] = frame["window_start"].dt.floor("5min")
        frame["temporal_split"] = frame["recording_day"].map(DAY_SPLITS)
        if frame["temporal_split"].isna().any():
            raise ValueError("Unexpected recording day; review the frozen protocol")
        targets = {
            "binary": ["binary_label"], "single": ["primary_class"],
            "multilabel": list(CLASSES),
        }[task]
        counts = frame.groupby("audio_path")[targets].transform("nunique", dropna=False)
        frame["conflicting_target"] = counts.gt(1).any(axis=1)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def coverage_targets(frame: pd.DataFrame, training: bool) -> list:
    """Define coverage goals without looking at predictions or test labels."""
    task = frame["task"].iloc[0]
    goals = []
    if task == "binary":
        minimum = 150 if training else 10
        for label in [0, 1]:
            mask = frame["binary_label"].eq(label)
            goals.append((f"binary_{label}", mask, minimum))
    elif task == "single":
        minimum = 20 if training else 10
        for label in CLASSES:
            mask = frame["primary_class"].eq(label)
            goals.append((label, mask, minimum))
    else:
        minimum = 20 if training else 10
        for label in CLASSES:
            mask = frame[label].eq(1)
            goals.append((label, mask, minimum))
        one_vessel = frame["n_vessels"].eq(1)
        multiple_vessels = frame["n_vessels"].gt(1)
        goals.append(("one_vessel", one_vessel, 50))
        goals.append(("multiple_vessels", multiple_vessels, 50))
        overlap_days = frame.loc[multiple_vessels, "recording_day"].unique()
        for day in sorted(overlap_days):
            mask = frame["recording_day"].eq(day) & multiple_vessels
            goals.append((f"overlap_day_{day}", mask, 1))
    return goals


def choose_rows(frame: pd.DataFrame, view: str, checker=None,
                base: list | None = None) -> tuple[pd.DataFrame, list]:
    """Choose rows in a fixed order, validating substitutes in the same split."""
    sorted_rows = frame.sort_values("audio_path")
    shuffled_rows = sorted_rows.sample(frac=1, random_state=SEED)
    order = shuffled_rows.index.tolist()
    selected, base = [], list(base or [])
    reasons = {}
    remaining = set(order) - set(base)
    day_counts = {}
    used_bins = set()

    goals = [] if view == "natural" else coverage_targets(frame, view == "training")
    steps = sorted(goals, key=lambda item: int(item[1].sum()))
    if view == "natural":
        steps.append(("uniform_validation", pd.Series(True, index=frame.index), VAL_ROWS))
    elif view == "training":
        steps.append(("day_bin_fill", pd.Series(True, index=frame.index), TRAIN_ROWS))

    spread = view != "natural"
    for position, (reason, mask, required) in enumerate(steps):
        # Coverage goals come first; an appended fill step uses an absolute row count.
        if position < len(goals):
            achieved = int(mask.loc[base + selected].sum())
            selection_count = len(selected) + max(0, required - achieved)
        else:
            selection_count = required
        preferred_day, failed_path = None, ""
        while len(selected) < selection_count:
            candidates = [i for i in order if i in remaining and mask.at[i]]
            if not candidates:
                break
            if spread:
                # Prefer unused bins, then the failed candidate's day, then day balance.
                # Stable sorting preserves the seeded order for ties.
                candidates.sort(key=lambda i: (
                    frame.at[i, "time_bin"] in used_bins,
                    preferred_day is not None and frame.at[i, "recording_day"] != preferred_day,
                    day_counts.get(frame.at[i, "recording_day"], 0),
                ))
            index = candidates[0]
            remaining.remove(index)
            row = frame.loc[index]
            if checker is not None and not checker(row, view, reason, failed_path):
                preferred_day, failed_path = row["recording_day"], row["audio_path"]
                continue
            selected.append(index)
            reasons[index] = reason
            day = row["recording_day"]
            day_counts[day] = day_counts.get(day, 0) + 1
            used_bins.add(row["time_bin"])
            preferred_day, failed_path = None, ""

    records = []
    if view != "natural":
        for goal, mask, required in coverage_targets(frame, view == "training"):
            achieved = int(mask.loc[base + selected].sum())
            records.append({"task": frame["task"].iloc[0],
                            "split": "train" if view == "training" else "val",
                            "goal": goal, "required": required, "achieved": achieved,
                            "met": achieved >= required})
    rows = frame.loc[selected].assign(view=view)
    rows["selection_reason"] = [reasons[i] for i in selected]
    return rows, records


def choose_sample(eligible: pd.DataFrame, reserved: pd.DataFrame,
                  checker=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select development views and retain the exact previous test membership."""
    parts, goals = [reserved], []
    for task in TASKS:
        task_rows = eligible["task"].eq(task)
        train = eligible.loc[task_rows & eligible["temporal_split"].eq("train")]
        val = eligible.loc[task_rows & eligible["temporal_split"].eq("val")]
        training_rows, targets = choose_rows(train, "training", checker)
        parts.append(training_rows)
        goals.extend(targets)
        natural_rows, _ = choose_rows(val, "natural", checker)
        parts.append(natural_rows)
        coverage_rows, targets = choose_rows(val, "coverage", checker, natural_rows.index.tolist())
        parts.append(coverage_rows)
        goals.extend(targets)
    return pd.concat(parts, ignore_index=True), pd.DataFrame(goals)


def validate_audio(row: pd.Series, view: str, reason: str, replaces: str) -> bool:
    """Read development WAVs only, retaining quiet audio and logging invalid files."""
    name = row["audio_path"]
    if name not in quality:
        info = archive.getinfo(PREFIX + "audio/" + name)
        examined.add(name)
        if audio_sizes.loc[list(examined)].sum() > MAX_AUDIO_BYTES:
            raise ValueError("Candidates including replacements exceed the approved 3 GB cap")
        path = DATA / info.filename
        if not path.is_file() or path.stat().st_size != info.file_size:
            archive.extract(info, DATA)
        entry = {"audio_path": name, "status": "invalid", "reason": ""}
        try:
            samples, rate = sf.read(path, always_2d=True)
            entry["frames"] = len(samples)
            entry["sample_rate"] = rate
            entry["channels"] = samples.shape[1]
            entry["duration"] = len(samples) / rate if rate else 0
            if rate != 32_000 or samples.shape[1] != 4 or len(samples) != 96_000:
                raise ValueError("Expected three seconds, four channels, and 32 kHz")
            if not np.isfinite(samples).all():
                raise ValueError("Nonfinite samples")
            peak = float(np.max(np.abs(samples)))
            entry["rms"] = float(np.sqrt(np.mean(samples ** 2)))
            entry["peak"] = peak
            entry["silent"] = peak == 0
            entry["clipping_fraction"] = float((np.abs(samples) >= 0.999).mean())
            entry["status"] = "valid"
        except (OSError, RuntimeError, ValueError) as error:
            entry["reason"] = str(error)
        quality[name] = entry
        if len(quality) % 100 == 0:
            print(f"Checked {len(quality)} development WAVs", flush=True)
    valid = quality[name]["status"] == "valid"
    attempts.append({
        "task": row["task"],
        "audio_path": name,
        "split": row["temporal_split"],
        "view": view,
        "selection_reason": reason,
        "replaces": replaces,
        "status": quality[name]["status"],
        "reason": quality[name]["reason"],
    })
    return valid


# Keep an immutable copy of the original pilot before changing any active exports.
OUTPUT.mkdir(parents=True, exist_ok=True)
for source, backup in [
    (OUTPUT / "pilot.csv", OUTPUT / "pilot-v1.csv"),
    (DATA / "pilot_selection.csv", DATA / "pilot-selection-v1.csv"),
    (DATA / "provenance.json", DATA / "provenance-v1.json"),
]:
    if not backup.exists():
        if not source.exists():
            raise FileNotFoundError("The original pilot is required to preserve test membership")
        shutil.copy2(source, backup)
previous = pd.read_csv(OUTPUT / "pilot-v1.csv")
reserved_keys = previous.loc[previous["split"].eq("test"), ["task", "audio_path", "official_split"]]

# RemoteZip uses HTTP ranges; metadata and archive sizes are read before any audio download.
with RemoteZip(URL, timeout=60, support_suffix_range=False) as archive:
    metadata = [PREFIX + "Portia_Metadata.json"] + [
        f"{PREFIX}{folder}/{split}.csv" for folder in TASKS.values() for split in SPLITS
    ]
    archive.extractall(DATA, members=metadata)
    manifests = read_manifests()
    manifests["split"] = manifests["temporal_split"]
    assert manifests.groupby("audio_path")["split"].nunique().le(1).all()

    # Exclude ambiguous targets, but keep the original rows in the full manifest.
    binary_valid = manifests["task"].eq("binary") & manifests["binary_label"].isin([0, 1])
    single_valid = manifests["task"].eq("single") & manifests["primary_class"].isin(CLASSES)
    multilabel_valid = (
        manifests["task"].eq("multilabel")
        & manifests[list(CLASSES)].isin([0, 1]).all(axis=1)
    )
    valid_targets = binary_valid | single_valid | multilabel_valid
    development = manifests["split"].ne("test")
    excluded = manifests.loc[
        development & (~valid_targets | manifests["conflicting_target"])
    ].copy()
    excluded["reason"] = np.where(
        excluded["conflicting_target"], "conflicting_target", "invalid_target",
    )
    eligible = manifests.loc[development & valid_targets & ~manifests["conflicting_target"]]
    duplicates = eligible.loc[eligible.duplicated(["task", "audio_path"])].assign(
        reason="duplicate",
    )
    eligible = eligible.drop_duplicates(["task", "audio_path"])
    # Duplicate source rows outside the selected holdout should not invalidate its merge.
    keys = ["task", "audio_path", "official_split"]
    reserved = reserved_keys.merge(manifests.drop_duplicates(keys), on=keys,
                                   validate="one_to_one")
    assert reserved["split"].eq("test").all() and len(reserved) == len(reserved_keys)
    reserved = reserved.assign(view="reserved_test", selection_reason="preserved_v1_test")

    # Flag metadata inconsistencies for review, without relabeling or removing quiet background.
    review = eligible.copy()
    invalid_count = review["n_vessels"].isna() | review["n_vessels"].lt(0)
    invalid_count |= review["n_vessels"].mod(1).ne(0)
    distance_columns = [name for name in review if name.endswith("_dist")]
    negative_distance = review[distance_columns].lt(0).any(axis=1)
    indicators = review[list(CLASSES)]
    inconsistent = indicators.sum(axis=1).gt(review["n_vessels"])
    inconsistent |= review["task"].eq("binary") & review["binary_label"].ne(
        review["n_vessels"].gt(0).astype(int),
    )
    review["reason"] = np.select([invalid_count, negative_distance, inconsistent],
                                ["invalid_vessel_count", "negative_distance",
                                 "indicator_count_mismatch"],
                                default="")
    review = review.loc[review["reason"].ne("")]

    inventory = pd.DataFrame([
        {"audio_path": Path(info.filename).name, "bytes": info.file_size}
        for info in archive.infolist()
        if info.filename.startswith(PREFIX + "audio/") and info.filename.endswith(".wav")
    ])
    audio_sizes = inventory.set_index("audio_path")["bytes"]
    planned, _ = choose_sample(eligible, reserved)
    names = planned["audio_path"].unique()
    audio_bytes = int(audio_sizes.loc[names].sum())
    transfer_bytes = 0
    for name in names:
        info = archive.getinfo(PREFIX + "audio/" + name)
        path = DATA / info.filename
        if not path.is_file() or path.stat().st_size != info.file_size:
            transfer_bytes += info.compress_size
    print(f"Planned {len(names)} unique WAVs: {audio_bytes / 1e9:.3f} GB selected; "
          f"about {transfer_bytes / 1e9:.3f} GB compressed transfer", flush=True)
    if audio_bytes > MAX_AUDIO_BYTES:
        raise ValueError("Planned audio exceeds the approved 3 GB cap; no audio downloaded")

    # Validate main validation before supplements; replacements follow the same fixed order.
    # Reserved test paths count toward the cap but are never opened or validated.
    quality, attempts = {}, []
    examined = set(reserved["audio_path"])
    pilot, goals = choose_sample(eligible, reserved, validate_audio)

# Accept the sample before overwriting active exports.
assert not pilot.duplicated(["task", "audio_path"]).any()
assert pilot.groupby("audio_path")["split"].nunique().le(1).all()
assert pilot.groupby("recording_day")["split"].nunique().le(1).all()
active_keys = ["task", "audio_path", "official_split"]
pd.testing.assert_frame_equal(
    pilot.loc[pilot["split"].eq("test"), active_keys]
    .sort_values(active_keys).reset_index(drop=True),
    reserved_keys.sort_values(active_keys).reset_index(drop=True), check_dtype=False,
)
audio_bytes = int(audio_sizes.loc[pilot["audio_path"].unique()].sum())
assert audio_bytes <= MAX_AUDIO_BYTES
audio_quality = pd.DataFrame(quality.values())
selection_attempts = pd.DataFrame(attempts)
audio_exclusions = selection_attempts.loc[selection_attempts["status"].eq("invalid")]
exclusions = pd.concat([excluded, duplicates, audio_exclusions], ignore_index=True)
summary = []
for (task, split, view), frame in pilot.groupby(["task", "split", "view"]):
    ids = pd.Series(dtype="string")
    if split != "test":
        ids = frame["all_mmsis"].dropna().astype(str).str.findall(r"\b\d{9}\b").explode()
    summary.append({"task": task, "split": split, "view": view, "rows": len(frame),
                    "days": frame["recording_day"].nunique(),
                    "time_bins": frame["time_bin"].nunique(),
                    "vessel_ids": ids.nunique() if split != "test" else None})
summary = pd.DataFrame(summary)

manifests.to_csv(OUTPUT / "manifests.csv", index=False)
pilot.to_csv(OUTPUT / "pilot.csv", index=False)
pilot[["audio_path", "task", "official_split", "split", "view"]].to_csv(
    DATA / "pilot_selection.csv", index=False,
)
inventory.to_csv(DATA / "audio_inventory.csv", index=False)
audio_quality.to_csv(OUTPUT / "audio_quality.csv", index=False)
exclusions.to_csv(OUTPUT / "exclusions.csv", index=False)
selection_attempts.to_csv(OUTPUT / "selection_attempts.csv", index=False)
review.to_csv(OUTPUT / "metadata_review.csv", index=False)
goals.to_csv(OUTPUT / "coverage_goals.csv", index=False)
summary.to_csv(OUTPUT / "sampling_summary.csv", index=False)
(DATA / "provenance.json").write_text(json.dumps({
    "record_id": RECORD, "protocol": "representative_v2", "sampling_split": "temporal",
    "seed": SEED, "train_rows_per_task": TRAIN_ROWS, "natural_val_rows_per_task": VAL_ROWS,
    "day_boundaries": {"train": TRAIN_DAYS, "val": VAL_DAYS, "test": TEST_DAYS},
    "time_bin_minutes": 5, "max_audio_bytes": MAX_AUDIO_BYTES,
    "selected_unique_audio": int(pilot["audio_path"].nunique()),
    "selected_audio_bytes": audio_bytes,
    "examined_unique_audio": len(examined),
    "invalid_audio": int(audio_quality["status"].eq("invalid").sum()),
    "exclusion_counts": exclusions["reason"].value_counts().to_dict(),
    "metadata_review_rows": len(review), "coverage": goals.to_dict(orient="records"),
    "sampling_summary": summary.astype(object).where(summary.notna(), None)
    .to_dict(orient="records"),
    "rules": {"training": "rare targets first, unused bins then day balance, seeded ties",
              "natural": "first 200 valid WAVs in seeded full-pool order; no rebalancing",
              "coverage": "remaining validation rows; gaps counted after natural selection",
              "replacements": "uniform next for natural; same target/day preferred otherwise",
              "audio": "3 seconds, 32 kHz, four channels, finite; silence/clipping flags only"},
    "ais_alignment": "unverified; source tracks and labeling rules not available locally",
}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(summary.to_string(index=False))
print("Unmet coverage goals:")
print(goals.loc[~goals["met"]].to_string(index=False))
print(f"Prepared {len(pilot)} memberships; {audio_bytes / 1e9:.3f} GB. Test audio not opened.")
