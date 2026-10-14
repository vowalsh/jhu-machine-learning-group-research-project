"""Make the figures used in presentation.md from the prepared PORTIA development data.

Run from the repo root after data-prep/prepare.py:

    uv run python data-presentation/make_figures.py

Only training and validation days are summarized. Audio examples and spectra use the
training views only, so no validation or reserved-test audio is opened here.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from matplotlib.patches import FancyBboxPatch
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
AUDIO = ROOT / "data" / "PORTIA_Clean_Release" / "audio"
FIGURES = ROOT / "data-presentation" / "figures"
CLASSES = ["tug", "pilot", "cargo", "passenger", "support", "tanker", "other"]
HYDROPHONE_CHANNELS = [0, 2, 3]
SPLIT_COLORS = {"train": "#4C72B0", "val": "#DD8452", "test": "#8C8C8C"}
TEXT_COLOR = "#222222"

FIGURES.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def hydrophone_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Average the Welch PSD of the three hydrophone channels and scale it to unit power."""
    samples, rate = sf.read(path, always_2d=True)
    frequencies, power = signal.welch(samples[:, HYDROPHONE_CHANNELS], fs=rate,
                                      nperseg=4096, axis=0)
    mean_power = power.mean(axis=1)
    return frequencies, mean_power / mean_power.sum()


# Load the prepared manifests; keep vessel IDs as strings.
id_columns = {"primary_mmsi": "string", "all_mmsis": "string"}
manifest = pd.read_csv(PROCESSED / "manifests.csv", dtype=id_columns)
pilot = pd.read_csv(PROCESSED / "pilot.csv", dtype=id_columns)

# Development rows only, with the same validity rules as data-prep/prepare.py.
development = manifest.loc[manifest["temporal_split"].isin(["train", "val"])]
valid_binary = development["task"].eq("binary") & development["binary_label"].isin([0, 1])
valid_single = development["task"].eq("single") & development["primary_class"].isin(CLASSES)
valid_multilabel = (
    development["task"].eq("multilabel") & development[CLASSES].isin([0, 1]).all(axis=1)
)
eligible = development.loc[
    (valid_binary | valid_single | valid_multilabel) & ~development["conflicting_target"]
].drop_duplicates(["task", "audio_path"])
single = eligible.loc[eligible["task"].eq("single")]
multilabel = eligible.loc[eligible["task"].eq("multilabel")]

# Figure: class support in the development pool for the two vessel-type tasks.
single_counts = single["primary_class"].value_counts().reindex(CLASSES)
multilabel_counts = multilabel[CLASSES].eq(1).sum()
order = single_counts.sort_values().index.tolist()
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
axes[0].barh(order, single_counts[order], color="#4C72B0")
axes[0].set(title="Single-vessel task: primary class", xlabel="Windows")
axes[1].barh(order, multilabel_counts[order], color="#55A868")
axes[1].set(title="Multi-label task: windows containing class", xlabel="Windows")
for ax, counts in [(axes[0], single_counts), (axes[1], multilabel_counts)]:
    for position, label in enumerate(order):
        ax.text(counts[label], position, f" {counts[label]:,}", va="center", fontsize=11)
    ax.set_xlim(0, counts.max() * 1.22)
fig.tight_layout()
fig.savefig(FIGURES / "class_counts.png")
plt.close(fig)
print("Single-task class counts:", single_counts.to_dict())
print("Multi-label positive counts:", multilabel_counts.to_dict())

# Figure: number of AIS-linked vessels per window, by task.
count_groups = ["1", "2", "3", "4+"]
vessel_rows = []
for task, frame in [("Single-vessel task", single), ("Multi-label task", multilabel)]:
    groups = pd.cut(frame["n_vessels"], bins=[0, 1, 2, 3, np.inf], labels=count_groups)
    shares = groups.value_counts(normalize=True).reindex(count_groups)
    vessel_rows.append(shares.rename(task))
vessel_shares = pd.DataFrame(vessel_rows).T
fig, ax = plt.subplots(figsize=(7.5, 4.2))
vessel_shares.plot.bar(ax=ax, rot=0, color=["#4C72B0", "#55A868"], width=0.75)
ax.set(xlabel="AIS-linked vessels in the window", ylabel="Share of windows")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIGURES / "vessels_per_window.png")
plt.close(fig)
print("Vessels-per-window shares:\n", vessel_shares.round(3))

# Figure: AIS distance from the hydrophones to the primary vessel.
fig, ax = plt.subplots(figsize=(7.5, 4.2))
bins = np.arange(0, 2.05, 0.05)
for label, frame, color in [("Single-vessel task", single, "#4C72B0"),
                            ("Multi-label task", multilabel, "#55A868")]:
    distances = frame["primary_dist"].dropna()
    ax.hist(distances, bins=bins, density=True, histtype="step", linewidth=2,
            color=color, label=f"{label} (median {distances.median():.2f} km)")
ax.set(xlabel="Distance to primary AIS vessel (km)", ylabel="Density")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIGURES / "primary_distance.png")
plt.close(fig)
print("Primary distance summary (km):")
print(pd.DataFrame({
    "single": single["primary_dist"].describe(),
    "multilabel": multilabel["primary_dist"].describe(),
}).round(3))

# Figure: recording days and the frozen temporal split (identities only, no test labels).
days = manifest.drop_duplicates("audio_path").groupby(["recording_day", "temporal_split"]).size()
days = days.reset_index(name="windows")
days["date"] = pd.to_datetime(days["recording_day"])
fig, ax = plt.subplots(figsize=(12, 3.8))
# A calendar axis keeps the days without released audio visible as gaps.
ax.bar(days["date"], days["windows"], width=0.8,
       color=days["temporal_split"].map(SPLIT_COLORS))
ax.set(xlabel="Recording day (2025)", ylabel="Unique 3-s windows")
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.tick_params(axis="x", rotation=45)
handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in SPLIT_COLORS.values()]
ax.legend(handles, ["train (our split)", "validation", "test (held out)"], frameon=False)
fig.tight_layout()
fig.savefig(FIGURES / "windows_per_day.png")
plt.close(fig)
print("Unique windows per temporal split:",
      days.groupby("temporal_split")["windows"].sum().to_dict())
print(f"Days with released audio: {len(days)}, "
      f"from {days['recording_day'].min()} to {days['recording_day'].max()}")

# Audio checks use the selected training views only.
training = pilot.loc[pilot["split"].eq("train") & pilot["view"].eq("training")]
binary_training = training.loc[training["task"].eq("binary")]
single_training = training.loc[training["task"].eq("single")]

# Per-channel RMS shows which of the four WAV channels carry hydrophone signals.
channel_rms = []
for name in binary_training["audio_path"]:
    samples, rate = sf.read(AUDIO / name, always_2d=True)
    channel_rms.append(np.sqrt(np.mean(samples ** 2, axis=0)))
channel_rms = pd.DataFrame(channel_rms, columns=[f"ch{i}" for i in range(4)])
print("Median per-channel RMS over binary training WAVs:")
print(channel_rms.median().round(4).to_dict())

# Figure: one background and one vessel-present training example, same colour scale.
background = binary_training.loc[binary_training["binary_label"].eq(0)].sort_values("audio_path")
vessel = binary_training.loc[
    binary_training["binary_label"].eq(1) & binary_training["n_vessels"].eq(1)
    & binary_training["primary_dist"].lt(0.3)
].sort_values("audio_path")
examples = [
    ("Background (no qualifying AIS vessel)", background.sample(1, random_state=42).iloc[0]),
    ("Vessel present", vessel.sample(1, random_state=42).iloc[0]),
]
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
spectrograms = []
for title, row in examples:
    samples, rate = sf.read(AUDIO / row["audio_path"], always_2d=True)
    frequencies, times, power = signal.spectrogram(samples[:, 0], fs=rate, nperseg=4096,
                                                   noverlap=3584)
    keep = (frequencies >= 20) & (frequencies <= 16_000)
    power_db = 10 * np.log10(np.maximum(power[keep], 1e-14))
    spectrograms.append((title, row, frequencies[keep], times, power_db))
# Share one colour scale so the two panels are directly comparable.
all_db = np.concatenate([item[4].ravel() for item in spectrograms])
low_db, high_db = np.percentile(all_db, [2, 100])
for ax, (title, row, frequencies, times, power_db) in zip(axes, spectrograms):
    mesh = ax.pcolormesh(times, frequencies, power_db, shading="auto", cmap="magma",
                         vmin=low_db, vmax=high_db)
    ax.set_yscale("log")
    ax.set_ylim(20, 16_000)
    if row["binary_label"] == 1:
        title += f": {row['primary_class']}, {row['primary_dist']:.2f} km"
    ax.set(title=title, xlabel="Time (s)")
    print(f"Spectrogram example {title}: {row['audio_path']}")
axes[0].set_ylabel("Frequency (Hz, log scale)")
fig.colorbar(mesh, ax=axes, label="Power (dB, uncalibrated)", pad=0.02)
fig.savefig(FIGURES / "spectrogram_examples.png")
plt.close(fig)

# Figure: median spectral shape for background vs vessel, and by vessel class.
binary_spectra = {0: [], 1: []}
for row in binary_training.itertuples(index=False):
    frequencies, spectrum = hydrophone_spectrum(AUDIO / row.audio_path)
    binary_spectra[int(row.binary_label)].append(spectrum)
class_spectra = {label: [] for label in CLASSES}
for row in single_training.itertuples(index=False):
    frequencies, spectrum = hydrophone_spectrum(AUDIO / row.audio_path)
    class_spectra[row.primary_class].append(spectrum)

keep = (frequencies >= 20) & (frequencies <= 16_000)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
for label, name, color in [(0, "Background", "#8C8C8C"), (1, "Vessel present", "#C44E52")]:
    median = np.median(binary_spectra[label], axis=0)
    axes[0].semilogx(frequencies[keep], 10 * np.log10(median[keep]), color=color, linewidth=2,
                     label=f"{name} (n={len(binary_spectra[label])})")
axes[0].set(title="Binary task, training view", xlabel="Frequency (Hz)",
            ylabel="Relative power (dB)")
axes[0].legend(frameon=False)
palette = plt.get_cmap("tab10")
for index, label in enumerate(CLASSES):
    median = np.median(class_spectra[label], axis=0)
    axes[1].semilogx(frequencies[keep], 10 * np.log10(median[keep]), color=palette(index),
                     linewidth=1.5, label=f"{label} (n={len(class_spectra[label])})")
axes[1].set(title="Single-vessel task, training view", xlabel="Frequency (Hz)")
axes[1].legend(frameon=False, fontsize=10, ncol=2)
fig.tight_layout()
fig.savefig(FIGURES / "median_spectra.png")
plt.close(fig)

# Print the numbers quoted on the slides: narrow low-frequency peaks and band differences.
background_median = np.median(binary_spectra[0], axis=0)
vessel_median = np.median(binary_spectra[1], axis=0)
low = frequencies < 400
peaks, _ = signal.find_peaks(10 * np.log10(background_median[low]), prominence=5)
print("Prominent background peaks below 400 Hz (Hz):", frequencies[low][peaks].round(1))
bands = [(20, 200), (200, 2000), (2000, 8000), (8000, 14000)]
for band_low, band_high in bands:
    in_band = (frequencies >= band_low) & (frequencies < band_high)
    difference = 10 * np.log10(vessel_median[in_band].sum() / background_median[in_band].sum())
    print(f"Vessel minus background, {band_low}-{band_high} Hz: {difference:+.1f} dB")
mid_band = (frequencies >= 200) & (frequencies < 2000)
for label in CLASSES:
    class_median = np.median(class_spectra[label], axis=0)
    difference = 10 * np.log10(class_median[mid_band].sum() / background_median[mid_band].sum())
    print(f"{label} minus background, 200-2000 Hz: {difference:+.1f} dB")

# Figure: data flow from sensors and AIS to labelled windows.
fig, ax = plt.subplots(figsize=(13, 4.2))
ax.set_xlim(0, 13.4)
ax.set_ylim(0, 4.2)
ax.axis("off")
boxes = [
    (0.1, 2.45, 3.8, 1.5, "#E8EEF7",
     "Hydrophones\n3 sensors, 32 kHz\nsaved as 3-s WAV files"),
    (0.1, 0.25, 3.8, 1.5, "#E8EEF7",
     "AIS reports\nevery minute, inside port,\nmoving (SOG > 1 knot)"),
    (5.0, 1.35, 3.6, 1.5, "#FFF4E0",
     "Temporal alignment\nWAVs within ±1 min\nof an AIS report"),
    (9.7, 1.35, 3.6, 1.5, "#E9F3EA",
     "Labelled window\npresence, vessel classes,\nAIS distance (km)"),
]
for x, y, width, height, color, text in boxes:
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.05",
                                facecolor=color, edgecolor="#555555", linewidth=1.2))
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=14,
            color=TEXT_COLOR, linespacing=1.4)
arrow = dict(arrowstyle="-|>", color="#555555", linewidth=1.5, mutation_scale=18)
ax.annotate("", xy=(4.95, 2.4), xytext=(3.95, 3.2), arrowprops=arrow)
ax.annotate("", xy=(4.95, 1.8), xytext=(3.95, 1.0), arrowprops=arrow)
ax.annotate("", xy=(9.65, 2.1), xytext=(8.65, 2.1), arrowprops=arrow)
fig.savefig(FIGURES / "pipeline.png")
plt.close(fig)

print(f"Saved figures to {FIGURES.relative_to(ROOT)}")
