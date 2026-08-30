# Exploratory Data Analysis of the SIIM-ISIC Melanoma Classification Dataset

> Markdown version of `EDA_Report.pdf`. Both files are kept in sync; the PDF remains the
> distributable copy, this file is the reviewable/diff-able one.

---

## 1. Introduction

Melanoma is a kind of skin cancer that starts in the melanocytes, the cells responsible for
producing pigment in the skin. The pigment is called melanin.

The exact cause of all melanomas isn't clear. Although melanoma is less common than some other
forms of skin cancer, it accounts for a significant proportion of skin cancer-related deaths.
Early and accurate detection is therefore an important area of research, and computer vision and
machine learning techniques have the potential to assist medical professionals in the
identification of suspicious skin lesions.

The purpose of this exploratory analysis is to gain a better understanding of the dataset before
applying machine learning / deep learning techniques. The analysis examines the structure and
quality of the metadata, the distribution of the target variable, patient-level information, and
characteristics of the available images. Particular attention is given to missing values,
duplicate observations and other factors that may influence the performance and reliability of a
predictive model.

---

## 2. Dataset Description

### Dataset size

| Dataset | Shape |
| --- | --- |
| Train | 33126 x 8 |
| Test | 10982 x 5 |

### Train and test structure

| Column | Train | Test |
| --- | :---: | :---: |
| `image_name` | ✓ | ✓ |
| `patient_id` | ✓ | ✓ |
| `sex` | ✓ | ✓ |
| `age_approx` | ✓ | ✓ |
| `anatom_site_general_challenge` | ✓ | ✓ |
| `diagnosis` | ✓ | n/a |
| `benign_malignant` | ✓ | n/a |
| `target` | ✓ | n/a |

### Missing values

| Column | Train missing | Test missing |
| --- | ---: | ---: |
| `image_name` | 0 | 0 |
| `patient_id` | 0 | 0 |
| `sex` | 65 | 0 |
| `age_approx` | 68 | 0 |
| `anatom_site_general_challenge` | 527 | 351 |
| `diagnosis` | 0 | n/a |
| `benign_malignant` | 0 | n/a |
| `target` | 0 | n/a |

The missing values in `sex` and `age_approx` are **dependent / correlated**.

### Target distribution

| Class | Rate |
| --- | ---: |
| Malignant | 1.8% |
| Benign | 98.2% |

The target distribution is **highly imbalanced**. This must be addressed during the training
phase. Possible solutions -> data augmentation, resampling, class weighting.

### Duplicates

There were no duplicate rows found in either dataset. However, a large number of patients have
multiple entries, which means the data is **not independent at the row level**. Visual inspection
confirmed that these repeated entries correspond to different lesions or imaging angles rather
than duplicate photos, but this still introduces a risk of **data leakage**, as a single patient's
images may share underlying characteristics (e.g. skin tone or mole color).

As a result, randomly splitting the dataset by row could allow information about a specific patient
to leak between training and validation sets. A **patient-level split** is recommended during
preprocessing.

The analysis shows that 33,126 images belong to only 2,056 patients.

### Unique values per column

| Column name | Unique values |
| --- | ---: |
| `image_name` | 33126 |
| `patient_id` | 2056 |
| `sex` | 2 |
| `age_approx` | 18 |
| `anatom_site_general_challenge` | 6 |
| `diagnosis` | 9 |
| `benign_malignant` | 2 |
| `target` | 2 |

### Outliers: age = 0

Found only 2 rows, with the same patient, who also has an `age = 10` entry.
Possibilities: data entry typo, or a 10-year gap between visits.

### Image resolution

Image resolutions vary across the datasets, ranging from 640x480 up to 6000x4000. Therefore all
images must be resized to a consistent input size during the preprocessing phase.

### Feature distributions

- **Sex:** 51.7% male, 48.3% female; malignant rate differs slightly.
- **Age:** with skewness = 0.081, the distribution is almost symmetric; malignant rate increases
  with age.
- **Anatomical site:** torso has the highest sample count in the dataset but a moderate malignant
  rate; sites with fewer samples like oral/genital show less reliable rate estimates and should be
  interpreted with caution.

---

## 3. Conclusion for the next steps

### 1. Missing data

The proportion of missing data is very low relative to the dataset size (`sex` at 0.196%,
`age_approx` at 0.205%, `anatom_site_general_challenge` at 1.59% in the training dataset, and
`anatom_site_general_challenge` at 3.196% in the test dataset), so dropping the affected rows would
have minimal impact on the overall dataset.

Alternatively, `sex` and `anatom_site_general_challenge` can be filled using the mode or an
`"unknown"` category, and `age_approx` can be filled using the median.

### 2. Class imbalance

As the data is highly imbalanced, we can use data augmentation, resampling or class weighting
(downsampling will cause a huge data loss).

### 3. Image preprocessing

As noted in the image resolution analysis, all images must be resized to a consistent input size
before being used for model training. Additionally, removing hair artifacts from images may
further improve model accuracy by reducing visual noise unrelated to the lesion itself.

Source: https://www.mdpi.com/2076-3417/16/4/1819

### 4. A patient-level split

Given that a large number of patients have multiple image entries, a patient-level split may be
used when creating training and validation sets to prevent information about a specific patient
from leaking between them.

### 5. Encoding

`image_name` and `patient_id` are identifier columns rather than predictive features. `image_name`
is required to load the corresponding image file for each observation, while `patient_id` may be
used to define patient-level train/validation splits. The remaining categorical columns (`sex`,
`anatom_site_general_challenge`, `diagnosis`, `benign_malignant`) need to be encoded into a numeric
format before being used in a model (e.g. OHE).
