# Supplementary Material: ECG Stress Detection Framework

This repository contains the source code and a representative dataset subset supporting the manuscript submitted for peer review. The project implements a stress detection pipeline using Genetic Algorithm (GA) optimized HRV features and a hybrid Deep Learning architecture (CTBFNet).

## 1. Dataset Availability & Access
**Current Status: Anonymized Review Subset (n=5)**

The full dataset collected for this study comprises **51 participants** (~1 GB of data). Due to the file size limits of the repository and to maintain the integrity of the double-blind review process, we have uploaded a **Representative Subset** containing data for **First 5 participants**.


The folder `Sample_Data_n5` contains:
* **Subjects:** 5 Participants (Anonymized IDs).
* **Conditions:** Full recordings for Baseline, Stroop, Mental Arithmetic Task (MAT), and Recovery phases.
* **Format:** Reconstructed continuous ECG signals (`.csv`).

### Requesting the Full Dataset
The complete dataset (51 participants) will be made publicly available upon acceptance of the manuscript. 

**For Reviewers:** If access to the full cohort data is strictly required for validation during the review process, please request it through the **Conference Program Chair** or **Area Chair**. This protocol ensures that the authors' identities remain concealed, preserving the double-blind nature of the review.

**Full dataset used is now uploaded**

## 2. Data Description
The dataset consists of raw ECG signals collected using a consumer-grade wearable device.

* [cite_start]**Acquisition Device:** Polar H10 Chest Strap[cite: 92].
* [cite_start]**Sampling Rate:** Signals were reconstructed from packets and interpolated to a uniform **130 Hz**.
* [cite_start]**Protocol:** The data is labeled according to a standardized four-phase protocol:
    1.  **Baseline:** Eyes-open resting state (5 mins).
    2.  **Stroop Task:** Cognitive load/Mild stress (5 mins).
    3.  **Mental Arithmetic Task (MAT):** Cognitive load/Severe stress (5 mins).
    4.  **Recovery:** Resting state (5 mins).

## License
Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

Copyright © 2026 Abu Saleh Khan, Areen Patil, Nipun Verma, Prakrititz Borah, Sakshi Arora.

This dataset is made available under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).

You are free to:

* Share — copy and redistribute the material in any medium or format.
* Adapt — remix, transform, and build upon the material.

Under the following terms:

* Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
* NonCommercial — You may not use the material for commercial purposes.
* No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

## Citation

If you use this dataset in academic research, publications, presentations, or other scholarly work, please cite the following paper:

Khan, A.S., Patil, A.R., Verma, N., Borah, P. and Arora, S. (2026). **Stress Detection from Consumer-Grade ECG Device Using GA Optimization and Segment-Wise Analysis**. In *Proceedings of the 19th International Joint Conference on Biomedical Engineering Systems and Technologies - BIOSIGNALS*; ISBN 978-989-758-802-0; ISSN 2184-4305, SciTePress, pp. 160–171. DOI: 10.5220/0014487500004070.

## Dataset Attribution

When using this dataset, please acknowledge the authors and cite the above publication.

The dataset is provided for academic and research purposes under the terms specified in the accompanying license. Users are responsible for ensuring that their use of the dataset complies with applicable ethical, privacy, and institutional requirements.

For the full license terms, see:
https://creativecommons.org/licenses/by-nc/4.0/
