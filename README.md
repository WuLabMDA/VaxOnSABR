# VaxOnSABR
**SARS-CoV-2 mRNA Vaccination Improved Survival in NSCLC Treated with Radiotherapy**

---

## Overview

Stereotactic ablative radiotherapy (SABR) is the standard of care for early-stage non-small cell lung cancer (NSCLC) and has demonstrated immunomodulatory properties that have prompted investigation of its combination with immune checkpoint inhibitors. SARS-CoV-2 mRNA vaccines are potent activators of innate immunity and have been associated with improved survival in several advanced cancers when administered in proximity to immunotherapy.

However, the potential interaction between peri-SABR vaccination and clinical outcomes in early-stage NSCLC remains poorly characterized. We performed a retrospective study evaluating the impact of peri-SABR SARS-CoV-2 mRNA vaccination on recurrence-free survival (RFS) and overall survival (OS) in a real-world cohort of early-stage NSCLC patients treated during the COVID-19 pandemic era, with independent validation in the I-SABR trial cohort.

To address immortal-time and selection biases inherent to observational vaccine studies, analyses were performed using time-varying Cox proportional hazards models with Simon-Makuch visualization, propensity score matching, and multivariate confounder adjustment.

This repository holds the code for the **I-SABR-SELECT** framework, as described in *[citation forthcoming]*.

---

## Repository Structure
```
VaxOnSABR/
├── Utils_fn/          # Utility functions
├── Main/              # Main figure scripts
├── Supp/              # Supplementary scripts
└── README.md
```

### 1. Utility Functions (`Utils_fn/`)
Functions supporting three types of analyses:
- Naive survival analysis
- Time-bias corrected analysis
- Selection bias corrected analysis

### 2. Main Scripts (`Main/`)
Scripts for generating the main figures:
- **Fig 1(a-b):** Time bias correction
- **Fig 1(c-d):** Vaccination timing during pandemic era
- **Fig 1(e-f):** Comprehensive bias analysis

### 3. Supplementary Codes (`Supp/`)
Scripts for generating supplementary figures:
- **S2:** Survival outcome by Flu Vaccine and Covid Infection
- **S3:** Survival outcome in the I-SABR trial full cohort analysis
- **Propensity match diagnostics:** Generated alongside Main Fig 1(d)

---

## Citation

> *Citation will be added upon publication.*

---

