# Data Repository Guide

This folder contains the sample datasets, metadata files, and databases used to validate and execute the **SecureCoating-Vision** system.

## Folder Structure

Please organize your input images and dataset as follows:

```
data/
├── README.md
├── sample_metadata.csv
├── raw/
│   ├── optical/        # High-resolution RGB images (.jpg or .png)
│   ├── thermal/        # Mid-infrared or LWIR thermal files (.tiff or .png)
│   └── 3d_profile/     # 16-bit heightmaps (.png or .bin)
└── processed/
    ├── aligned/        # Aligned/warped 5-channel numpy arrays (.npy)
    └── masks/          # Defect ground truth segmentation masks (.png)
```

## LIBAD external validation set

The official LIBAD release is **not** stored in git (~4.84 GB, CC BY 4.0). After accepting the dataset terms, extract it to:

```
data/libad/LIBAD/
data/libad/splits/
```

See `scripts/download_libad.py --probe` and [docs/libad_validation_extension.md](../docs/libad_validation_extension.md). The script will not start the 4.84 GB transfer without `--download --accept-license`. Fixture protocol tests do not require this download and must not be reported as paper-comparable LIBAD numbers.


The primary metadata spreadsheet is `sample_metadata.csv`. It bridges multi-source files together and registers their inspection labels.

| Column | Description | Example |
| :--- | :--- | :--- |
| `batch_id` | Unique ID of the manufacturing batch | `BATCH_2026_06A` |
| `part_id` | Unique identifier for the individual parts | `PART_2026_06A_001` |
| `optical_path` | Relative path to high-res optical image | `raw/optical/part_001.png` |
| `thermal_path` | Relative path to thermal image | `raw/thermal/part_001.png` |
| `height_path` | Relative path to 3D profile map | `raw/3d_profile/part_001.png` |
| `has_defect` | Boolean flag (0 = Pass, 1 = Fail) | `1` |
| `defect_class` | Primary defect category label | `delamination` |
| `timestamp` | Execution local time | `2026-06-27T08:00:00Z` |
