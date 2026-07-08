# Model Training Scores

All models use YOLOv8m. Scores are from the best epoch on the validation split.  
Classes trained: `meteorite`, `fusion_crust`, `scale_reference` (regmaglypts and metal_flake excluded — too few examples).

---

## 2026-07-06 — Four-model training run

All four models: negatives included, sectioned excluded.

| Model | Dataset | Images | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|---|---|
| `in_situ_no_mock` | In-situ only, no mock | 133 | 0.729 | 0.557 | — | — |
| `in_situ_inc_mock` | In-situ only, inc. mock | 248 | 0.732 | 0.555 | — | — |
| `all_meteorites_no_mock` | All, no mock | 946 | 0.986 | 0.929 | — | — |
| `all_meteorites_inc_mock` | All, inc. mock | 946 | 0.981 | 0.933 | — | — |

### Per-class scores (final validation)

| Model | meteorite mAP50 | fusion_crust mAP50 | scale_reference mAP50 |
|---|---|---|---|
| `in_situ_no_mock` | 0.862 | 0.673 | 0.652 |
| `in_situ_inc_mock` | 0.830 | 0.635 | 0.731 |
| `all_meteorites_no_mock` | 0.994 | 0.980 | 0.983 |
| `all_meteorites_inc_mock` | 0.992 | 0.969 | 0.980 |

**Notes:**
- In-situ models score lower due to small dataset size (~133 images total after val split of 19).
- All-meteorites models benefit from ~946 images; mock inclusion has minimal effect at this scale.

---

## 2026-06-xx — v1.1 (current deployed model)

Run name: `meteorite_detector`  
Filters: all meteorites, sectioned excluded, no mock filter, no negatives.

| mAP50 | mAP50-95 | Precision | Recall | Images (approx.) |
|---|---|---|---|---|
| 0.986 | 0.939 | 0.976 | 0.975 | ~830 |

Best epoch: 90 of 100.

---

## 2026-06-xx — v1.0

Run name: `meteorite_detector` (earlier run)  
Filters: all meteorites, no filters applied.

| mAP50 | mAP50-95 |
|---|---|
| 0.973 | 0.917 |
