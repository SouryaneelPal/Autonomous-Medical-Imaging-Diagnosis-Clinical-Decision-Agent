"""
Synthetic RSNA-schema data generator -- Phase 1.0.

*** SYNTHETIC DATA -- NOT CLINICAL EVIDENCE ***

Every pixel, label, and patient this script produces is randomly
generated. Nothing here represents a real patient or a real clinical
case, and NO metric computed against its output means anything
clinically. It exists so the Phase 1 evaluation harness (locked split,
sensitivity/specificity/AUROC, calibration, subgroup bias analysis, the
Model Card generator) can be built and mechanically verified against
data that is schema-correct for the real target dataset, ahead of the
real RSNA Pneumonia Detection Challenge data landing in this
environment -- see Strategic_Startup_Roadmap.pdf, Phase 1.

Schema verified against RSNA's own dataset description
(rsna.org: "2018 RSNA Pneumonia Detection Challenge Dataset Description")
and the real competition CSV files:

  stage_2_train_labels.csv: patientId,x,y,width,height,Target
    - Target 0/1 (pneumonia/lung-opacity absent/present)
    - x/y/width/height blank when Target=0
    - a patientId has MULTIPLE ROWS when it has multiple bounding boxes

  stage_2_detailed_class_info.csv: patientId,class
    - class in {"Normal", "Lung Opacity", "No Lung Opacity / Not Normal"}
    - row-aligned 1:1 with stage_2_train_labels.csv: same patientId,
      same row order, same per-patient duplication

  DICOM tags RSNA added on top of the source NIH CXR8 images:
    PatientAge (DICOM AS format, e.g. "064Y"), PatientSex ("M"/"F"),
    ViewPosition ("PA"/"AP")

  Real cohort shape this generator matches (publicly documented
  proportions, not exact real values): 1024x1024 single-channel images,
  ~54.2%/45.8% PA/AP (16,248 / 13,752 of 30,000), ~31%/40%/29%
  Lung Opacity/Normal/No Lung Opacity-Not Normal.

Every generated DICOM's PatientName and InstitutionName are set to an
unmistakable watermark string, and a SYNTHETIC_DATA_README.txt is
written alongside the CSVs, so this can never be mistaken for a real
export even if a directory gets copied somewhere out of context.
"""
from __future__ import annotations

import argparse
import csv
import logging
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

logger = logging.getLogger("medagent.data.synthetic_rsna_generator")

SYNTHETIC_WATERMARK = "SYNTHETIC-NOT-REAL-DATA"

_CLASS_TO_TARGET = {
    "Normal": 0,
    "No Lung Opacity / Not Normal": 0,
    "Lung Opacity": 1,
}
# Matches RSNA's publicly documented class proportions (~31/40/29).
_CLASS_WEIGHTS = {"Lung Opacity": 0.31, "Normal": 0.40, "No Lung Opacity / Not Normal": 0.29}
# Matches RSNA's official 16,248 PA / 13,752 AP split of 30,000 images.
_VIEW_POSITION_WEIGHTS = {"PA": 16248 / 30000, "AP": 13752 / 30000}


def _sample_age(rng: np.random.Generator) -> int:
    """Adult-skewed age distribution, clipped to a plausible clinical
    range -- NOT RSNA's real age distribution (not published at this
    granularity), just realistic enough to exercise all four age bands
    defined in evaluation/demographics.py."""
    age = int(rng.normal(loc=55, scale=18))
    return max(1, min(age, 95))


def _sample_sex(rng: np.random.Generator) -> str:
    # Real RSNA PatientSex values are M/F only; a small "O" fraction is
    # added here deliberately -- NOT a claim about the real dataset's
    # demographics -- purely so this generator's output exercises
    # state.py's full Literal["M","F","O"] and stratified.py's low-n
    # handling for a rare subgroup at least once.
    return rng.choice(["M", "F", "O"], p=[0.485, 0.485, 0.03])


def _sample_class(rng: np.random.Generator) -> str:
    return rng.choice(list(_CLASS_WEIGHTS.keys()), p=list(_CLASS_WEIGHTS.values()))


def _sample_view_position(rng: np.random.Generator) -> str:
    return rng.choice(list(_VIEW_POSITION_WEIGHTS.keys()), p=list(_VIEW_POSITION_WEIGHTS.values()))


def _sample_boxes(rng: np.random.Generator, image_size: int, num_boxes: int) -> list[tuple[float, float, float, float]]:
    """1-3 plausible-shaped bounding boxes within the image bounds --
    "plausible" as in "a reasonable rectangle inside the frame", not
    "clinically located over a real opacity"."""
    boxes = []
    for _ in range(num_boxes):
        w = rng.uniform(image_size * 0.1, image_size * 0.35)
        h = rng.uniform(image_size * 0.1, image_size * 0.35)
        x = rng.uniform(0, image_size - w)
        y = rng.uniform(0, image_size - h)
        boxes.append((round(float(x), 1), round(float(y), 1), round(float(w), 1), round(float(h), 1)))
    return boxes


def _write_dicom(
    path: Path,
    patient_id: str,
    age: int,
    sex: str,
    view_position: str,
    image_size: int,
    rng: np.random.Generator,
) -> None:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\x00" * 128)
    ds.PatientID = patient_id
    ds.PatientAge = f"{age:03d}Y"  # DICOM AS (age string) format
    ds.PatientSex = sex
    ds.ViewPosition = view_position
    ds.Modality = "CR"
    ds.BodyPartExamined = "CHEST"
    ds.PatientName = SYNTHETIC_WATERMARK
    ds.InstitutionName = SYNTHETIC_WATERMARK
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = image_size
    ds.Columns = image_size
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0

    pixels = (rng.random((image_size, image_size)) * 255).astype("uint8")
    ds.PixelData = pixels.tobytes()

    ds.save_as(str(path), enforce_file_format=True)


def generate(
    output_dir: str | Path,
    num_patients: int = 300,
    image_size: int = 1024,
    seed: int = 42,
) -> None:
    """Generates `num_patients` synthetic patients into `output_dir`,
    laid out exactly like a real RSNA download: stage_2_train_images/,
    stage_2_train_labels.csv, stage_2_detailed_class_info.csv."""
    output_dir = Path(output_dir)
    images_dir = output_dir / "stage_2_train_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    label_rows: list[dict] = []
    class_rows: list[dict] = []
    class_counts: Counter[str] = Counter()
    view_counts: Counter[str] = Counter()
    sex_counts: Counter[str] = Counter()

    for _ in range(num_patients):
        patient_id = str(uuid.uuid4())
        age = _sample_age(rng)
        sex = _sample_sex(rng)
        view_position = _sample_view_position(rng)
        klass = _sample_class(rng)
        target = _CLASS_TO_TARGET[klass]

        class_counts[klass] += 1
        view_counts[view_position] += 1
        sex_counts[sex] += 1

        _write_dicom(images_dir / f"{patient_id}.dcm", patient_id, age, sex, view_position, image_size, rng)

        if target == 0:
            label_rows.append({"patientId": patient_id, "x": "", "y": "", "width": "", "height": "", "Target": 0})
            class_rows.append({"patientId": patient_id, "class": klass})
        else:
            num_boxes = int(rng.integers(1, 4))  # 1-3, matching the real dataset's multi-box patients
            for x, y, w, h in _sample_boxes(rng, image_size, num_boxes):
                label_rows.append({"patientId": patient_id, "x": x, "y": y, "width": w, "height": h, "Target": 1})
                class_rows.append({"patientId": patient_id, "class": klass})

    labels_path = output_dir / "stage_2_train_labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["patientId", "x", "y", "width", "height", "Target"])
        writer.writeheader()
        writer.writerows(label_rows)

    class_info_path = output_dir / "stage_2_detailed_class_info.csv"
    with class_info_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["patientId", "class"])
        writer.writeheader()
        writer.writerows(class_rows)

    (output_dir / "SYNTHETIC_DATA_README.txt").write_text(
        "*** SYNTHETIC DATA -- NOT CLINICAL EVIDENCE ***\n\n"
        f"Generated by data/synthetic_rsna_generator.py (seed={seed}, num_patients={num_patients}).\n"
        "Every image, label, and demographic value in this directory is randomly generated.\n"
        "It matches the RSNA Pneumonia Detection Challenge's file/CSV/DICOM schema for\n"
        "pipeline-testing purposes only. No metric computed against this data reflects\n"
        "real clinical performance -- see Strategic_Startup_Roadmap.pdf, Phase 1.\n"
    )

    logger.info(
        "Generated %d synthetic patients, %d label rows -> %s",
        num_patients, len(label_rows), output_dir,
    )
    logger.info("class distribution: %s", dict(class_counts))
    logger.info("view_position distribution: %s", dict(view_counts))
    logger.info("sex distribution: %s", dict(sex_counts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="data/synthetic_rsna", help="RSNA-shaped output directory")
    parser.add_argument("--num-patients", type=int, default=300)
    parser.add_argument("--image-size", type=int, default=1024, help="Square image side length in pixels")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    generate(args.output_dir, num_patients=args.num_patients, image_size=args.image_size, seed=args.seed)


if __name__ == "__main__":
    main()
