# PORTIA dataset notes for module 7

This note covers the dataset background, reasonable assumptions, hypothesized predictive power, and why the dataset matters for the module 7 prompt. It does not pick the group research question. Jose is doing more data work and the research question. `data-prep/` and `eda/` already cover preparation and the first look.

## Background

The [paper abstract](https://doi.org/10.1038/s41597-026-08384-9) by Fernandez Garcia et al. was published in Scientific Data on 2 October 2026. The page says this is an early version and is subject to further edits. PORTIA comes from a 26-day campaign in the Port of Valencia, Spain. It pairs passive acoustic recordings with synchronized AIS metadata. The paper's full description has 260,932 three-second audio windows.

The paper presents PORTIA as a benchmark for vessel detection, vessel-type classification, and proximity estimation in a dense port. Its three benchmarks are Binary Detection with 36,148 samples, Single-Vessel Classification with 39,125 samples, and Multi-Label Monitoring with 34,996 samples. The benchmarks use MMSI-aware splitting to limit identity leakage and use stratified sampling. The work was funded by SMAUG, European Commission grant 101121129.

This repo does not download the paper release. `data-prep/prepare.py` uses [Zenodo record 19204539](https://zenodo.org/records/19204539), also identified by [DOI 10.5281/zenodo.19204539](https://doi.org/10.5281/zenodo.19204539). That record was published on 1 May 2026, collected in May 2025, and available on 24 March 2026. Its page reports 98,297 audio windows, with 36,778 binary-labeled samples, 39,996 single-vessel samples, and 34,996 multilabel samples.

The May record lists tug, pilot, cargo, passenger, support, tanker, and other as classes. In the single-vessel subset, pilot has 11,661 samples and tug has 7,784. Multilabel co-occurrence examples include tug with 22,022 samples and pilot with 16,910. Labels come from AIS spatiotemporal proximity. Vessel-sensor distance ranges from 0.02 km to 1.98 km, with a mean of 0.6 km and a median of 0.48 km. The page describes overlapping sources and class imbalance as real traffic, not a cleaned balance. The record has one file, `PORTIA.zip`, at 76.7 GB with md5 `8f7ba84acc7260bcaecff25fdcf4f33d`.

A later release, [Zenodo v2 record 21409381](https://zenodo.org/records/21409381), also identified by [DOI 10.5281/zenodo.21409381](https://doi.org/10.5281/zenodo.21409381), was published on 17 July 2026. It matches the paper's benchmark counts, not the May zip. It reports 95,022 unique audio windows and 110,269 benchmark memberships, so some windows sit in more than one task.

The v2 page describes three-second windows at 32 kHz with four channels and three hydrophones. Channel 1 is unused. The description says 32-bit PCM, while the verification list says 16-bit PCM. Both are reported here without choosing between them. Labels are AIS co-occurrence, not exhaustive manual acoustic annotation. The page names vessels without AIS, position error, and vessels outside the annotation area as reasons a label can disagree with what is audible.

## Published count sets

| Source | Audio windows | Binary Detection | Single-Vessel Classification | Multi-Label Monitoring |
| --- | ---: | ---: | ---: | ---: |
| October paper | 260,932 in the full description | 36,148 | 39,125 | 34,996 |
| May record 19204539 | 98,297 | 36,778 | 39,996 | 34,996 |
| July v2 | 95,022 unique | 36,148 | 39,125 | 34,996 |

These are published record-page counts, not row counts from a local CSV. We did not re-download the zip.

## Hypothesized predictive power

The working hypothesis is that the audio windows carry predictive signal for vessel presence, vessel type, and proximity. These are the outcomes named by the paper's detection, classification, and proximity benchmarks.

## Assumptions

1. **Assumption:** Do not put the paper's 36,148, 39,125, and 34,996 on a slide about this repo's tables until someone checks the CSVs from record 19204539. The May page and the paper disagree.
2. **Assumption:** An AIS label is not proof that the vessel is the sound in the window. `eda/eda.ipynb` already says labels come from AIS, not verified audible sources. Do not restate the plots.
3. **Assumption:** Jose's notebook says the official splits can share recordings. The paper says splits are MMSI-aware. Those are different claims, meaning same file versus same ship. Do not call either wrong.
4. **Assumption:** Class imbalance on the May page is the port's traffic, not a bug in the sample.
5. **Assumption:** Channel layout and sample format on the v2 page are not facts about the May zip until someone opens a wav from record 19204539.

## Why it matters

Large labeled sets from a real busy port are scarce. The paper is days old and still an early version. The midpoint prompt wants a dataset that does not already have a guaranteed answer.

## What this file does not do

This file does not set a research question or choose a model. It makes no change to `prepare.py` or the notebook.
