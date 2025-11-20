# Supplementary Material: ECG Stress Detection Framework

This repository contains the source code and a representative dataset subset supporting the manuscript submitted for peer review. The project implements a stress detection pipeline using Genetic Algorithm (GA) optimized HRV features and a hybrid Deep Learning architecture (CTBFNet).

## 1. Dataset Availability & Access
**Current Status: Anonymized Review Subset (n=5)**

The full dataset collected for this study comprises **51 participants** (~1 GB of data). Due to the file size limits of the repository and to maintain the integrity of the double-blind review process, we have uploaded a **Representative Subset** containing data for **5 randomly selected participants**.

The folder `Sample_Data_n5` contains:
* **Subjects:** 5 Participants (Anonymized IDs).
* **Conditions:** Full recordings for Baseline, Stroop, Mental Arithmetic Task (MAT), and Recovery phases.
* **Format:** Reconstructed continuous ECG signals (`.csv`).

### Requesting the Full Dataset
The complete dataset (51 participants) will be made publicly available upon acceptance of the manuscript. 

**For Reviewers:** If access to the full cohort data is strictly required for validation during the review process, please request it through the **Conference Program Chair** or **Area Chair**. This protocol ensures that the authors' identities remain concealed, preserving the double-blind nature of the review.

## 2. Data Description
The dataset consists of raw ECG signals collected using a consumer-grade wearable device.

* [cite_start]**Acquisition Device:** Polar H10 Chest Strap[cite: 92].
* [cite_start]**Sampling Rate:** Signals were reconstructed from packets and interpolated to a uniform **130 Hz**[cite: 93, 101].
* [cite_start]**Protocol:** The data is labeled according to a standardized four-phase protocol[cite: 61]:
    1.  **Baseline:** Eyes-open resting state (5 mins).
    2.  **Stroop Task:** Cognitive load/Mild stress (5 mins).
    3.  **Mental Arithmetic Task (MAT):** Cognitive load/Severe stress (5 mins).
    4.  **Recovery:** Resting state (5 mins).

## 3. Repository Structure
```text
/
├── main_pipeline.py        # End-to-end training script (Preprocessing -> GA -> Model)
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── Sample_Data_n5.zip      # Zipped subset of 5 participants (Unzip before use)
