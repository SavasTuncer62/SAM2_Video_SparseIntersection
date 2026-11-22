# Sparse CT Slices as Video: SAM2 Memory Propagation for Pneumothorax Segmentation

Code for the manuscript:

> **Treating Sparse Computed Tomography Slices as Video: Memory-Based Propagation with Segment Anything Model 2 for Pneumothorax Segmentation and Its Dependence on the Supervised Baseline**
> Savaş Tunçer, Mustafa Yıldırım, Özal Yıldırım. *Manuscript under review.*

The sparse axial CT slices of each patient are treated as the frames of a video. An expert prompt on one key slice is propagated forwards and backwards with the memory of Segment Anything Model 2 (SAM2), without fine-tuning. Propagation is compared with a weak and a strong supervised model trained on the same data.

| Code | Method | Expert input |
|---|---|---|
| B0 | Strong supervised model, each slice independently | none |
| B1 | B0 + SAM2 image-mode fusion | none |
| B2 | Copy the key-slice reference mask (lower bound) | mask |
| P1 | SAM2 video propagation from a reference mask | mask |
| P2 | SAM2 video propagation from a bounding box | box |
| P2N | SAM2 video propagation from a box plus an interior point | box + 1 click |
| H | Hybrid fusion of P1 with the strong model | mask |
| P3 | Automatic propagation from the most confident model mask | none |
| P4 / P4S | Hard / soft evidence gate between propagation and the model | none |

## Repository contents

```
study2_sam2_sparse_slice_propagation/
├── pnx_ortak.py                                 shared helper module (data split, U-Net and SAM2 inference, metrics)
├── requirements.txt
└── notebooks/
    ├── 1_sparse_slice_video_propagation.ipynb   all methods above, validation selection, test evaluation, statistics, figures
    ├── 2_validation_tuned_weak_baseline.ipynb   weak baseline with post-processing selected on validation patients
    ├── 3_slice_order_check.ipynb                objective pre-check of slice order and a review form for a radiologist
    └── additional_analyses/
        ├── A1_key_slice_and_registration_baseline.ipynb   middle / random key slice, optical-flow registration baseline, body-centred hemithorax split
        ├── A2_sparsification_on_open_dense_ct.ipynb       controlled sparsification of complete CT series from ReXGroundingCT / CT-RATE
        └── A3_representative_example_patient.ipynb        predefined selection of a representative example patient
```

The notebooks in `additional_analyses/` are sensitivity analyses. They reuse the trained weights and do not retrain any model.

## Running order

All notebooks are written for Google Colab with a T4 or L4 GPU and Google Drive as storage. The first cells install the packages, clone SAM2, download the SAM2.1 checkpoints (Hiera-Small and Hiera-Base+) and write `pnx_ortak.py` to the runtime disk.

1. Train the supervised models with the segmentation-backbone notebook of the companion repository, or provide equivalent weights (see *Expected data layout*).
2. `1_sparse_slice_video_propagation.ipynb` runs all methods. The SAM2 size, the hybrid rule and the gate threshold are selected on the validation patients, and the test patients are evaluated once with the selected settings.
3. `2_validation_tuned_weak_baseline.ipynb` re-evaluates the weak baseline with validation-tuned post-processing on the same test slices.
4. `3_slice_order_check.ipynb` (optional) produces a pre-check of the file-number slice order and a PDF/CSV form for radiologist review.
5. Optional: the notebooks in `additional_analyses/`.

## Expected data layout

The in-house dataset is **not included** (see *Data availability*). The notebooks expect the following files on Google Drive:

| Variable | Default path | Content |
|---|---|---|
| `RAW_DIR` (in `pnx_ortak.py`) | `/content/drive/MyDrive/CT_goruntuler_anon` | One folder per patient with `N.jpg` slices and `N.png` label maps (0 background, 1 pneumothorax, 2 lung). The file number defines the slice order |
| `DRIVE_BACKUP_DIR` (in `pnx_ortak.py`) | `/content/drive/MyDrive/pnx_artifacts` | Weights of the weak baseline, a MobileNetV2 U-Net trained on 224 × 224 patches (`pnomotoraks_unet.pth`) |
| `GUCLU_YOL` | `/content/drive/MyDrive/pnx_model/unet_resnet34_pnomotoraks.pth` | Weights of the strong single-task ResNet34 U-Net |

Only lower-case `.jpg`/`.png` pairs are loaded. Other extensions are skipped and listed.

`A2_sparsification_on_open_dense_ct.ipynb` downloads complete chest CT series from the gated Hugging Face datasets [ReXGroundingCT](https://huggingface.co/datasets/rajpurkarlab/ReXGroundingCT) and [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) (CC BY-NC-SA 4.0). Accept the access conditions on both dataset pages and store a read-access token as the Colab secret `HF_TOKEN`. Never write the token into the notebook.

## Software

- Python 3, PyTorch (Colab default)
- `segmentation_models_pytorch==0.5.0`
- SAM2 from https://github.com/facebookresearch/sam2 with the `sam2.1_hiera_small.pt` and `sam2.1_hiera_base_plus.pt` checkpoints
- OpenCV (including the DIS optical flow used for the registration baseline), SciPy, scikit-learn, pandas, matplotlib, Pillow
- `nibabel`, `huggingface_hub`, `scikit-image` (only for `A2_sparsification_on_open_dense_ct.ipynb`)

See `requirements.txt`. The installation cells in the notebooks pin the same versions.

## Reproducibility

- Patients are split at patient level with a fixed seed (42): 111 training, 17 validation and 43 test patients.
- All selections (SAM2 size, hybrid rule, gate threshold, weak-model post-processing) are made on the validation patients only.
- Python, NumPy and PyTorch seeds are fixed at 42.
- The notebooks write slice-level results as CSV files to their output folders. Some summary statistics and the final figures of the manuscript were computed from these CSV files.

## Notes

- Comments and variable names in the notebooks are partly in Turkish (ASCII characters only).
- Expert annotation is simulated with the reference mask of the key slice.
- The code is a research prototype and is not intended for clinical use.

## Data availability

The in-house CT dataset was collected retrospectively at a university hospital with ethics committee approval and is not publicly available because of patient privacy. It can be requested from the corresponding author, subject to ethics committee approval. The open datasets used in `A2_sparsification_on_open_dense_ct.ipynb` are available from their Hugging Face pages under their access terms.

## Citation

The citation will be added once the article is published.

## Contact

Savaş Tunçer (corresponding author), savastuncer62@gmail.com
