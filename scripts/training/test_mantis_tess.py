
"""
MODEL B - MANTIS ON REAL TESS DATA

Purpose:
    Verify pretrained Mantis-8M feature extraction
    using actual TESS phase-folded light curves.

Uses:
    - 16 training samples only
    - local and global flux views
    - pretrained frozen MantisV1 encoder

Does NOT:
    - train a classifier
    - evaluate validation data
    - evaluate test data
    - modify Model A checkpoints

Run:
    python -u scripts/training/test_mantis_tess.py
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mantis.architecture import MantisV1
from mantis.trainer import MantisTrainer


ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "full_dataset"

INPUT = DATA / "full_cnn_views.npz"

SPLITS = DATA / "cnn_split_indices.npz"

OUTPUT = DATA / "mantis_model_b"

CHECKPOINT = "paris-noah/Mantis-8M"

DEVICE = "cuda"

SAMPLE_COUNT = 16

SEED = 42

LENGTH = 512


def prepare_view(data):
    """
    Convert a TESS view to Mantis format.

    Expected input:
        (samples, time_points, 2)

    Channel 0:
        flux in the existing CNN dataset

    Channel 1:
        coverage mask

    This initial pilot uses flux only.

    The asinh transformation follows the
    existing CNN preprocessing convention.

    Output:
        (samples, 1, 512)
    """

    if data.ndim != 3 or data.shape[2] != 2:
        raise ValueError(
            f"Unexpected view shape: {data.shape}"
        )

    flux = np.asarray(
        data[:, :, 0],
        dtype=np.float32,
    )

    if not np.isfinite(flux).all():
        raise ValueError(
            "Non-finite flux values detected"
        )

    transformed = np.arcsinh(
        flux / 5.0
    ).astype(np.float32)

    tensor = torch.from_numpy(
        transformed[:, None, :]
    )

    resized = F.interpolate(
        tensor,
        size=LENGTH,
        mode="linear",
        align_corners=False,
    )

    result = resized.numpy()

    if not np.isfinite(result).all():
        raise ValueError(
            "Non-finite resized input"
        )

    return result


def main():
    print("=" * 70)
    print("MODEL B - MANTIS ON REAL TESS DATA")
    print("=" * 70)

    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)

    if not SPLITS.is_file():
        raise FileNotFoundError(SPLITS)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable"
        )

    print("GPU:", torch.cuda.get_device_name(0))

    print("\nLoading TESS dataset...")

    with np.load(
        INPUT,
        allow_pickle=False,
    ) as data:

        required = (
            "x_local",
            "x_global",
        )

        for name in required:
            if name not in data.files:
                raise KeyError(
                    f"Missing dataset array: {name}"
                )

        local_shape = data["x_local"].shape
        global_shape = data["x_global"].shape

        if local_shape != (1918, 101, 2):
            raise ValueError(
                f"Unexpected local shape: {local_shape}"
            )

        if global_shape != (1918, 201, 2):
            raise ValueError(
                f"Unexpected global shape: {global_shape}"
            )

        print("Local dataset:", local_shape)
        print("Global dataset:", global_shape)

        with np.load(
            SPLITS,
            allow_pickle=False,
        ) as split:

            train_idx = np.asarray(
                split["train_idx"],
                dtype=np.int64,
            )

        if train_idx.size != 1335:
            raise ValueError(
                "Unexpected training split size"
            )

        if (
            train_idx.min() < 0
            or train_idx.max() >= local_shape[0]
            or np.unique(train_idx).size != train_idx.size
        ):
            raise ValueError(
                "Invalid training indices"
            )

        rng = np.random.default_rng(
            SEED
        )

        selected = np.sort(
            rng.choice(
                train_idx,
                size=SAMPLE_COUNT,
                replace=False,
            )
        )

        print(
            "Selected training examples:",
            SAMPLE_COUNT,
        )

        print(
            "Selected indices:",
            selected.tolist(),
        )

        local_data = np.asarray(
            data["x_local"][selected],
            dtype=np.float32,
        )

        global_data = np.asarray(
            data["x_global"][selected],
            dtype=np.float32,
        )

    print("\nPreparing Mantis inputs...")

    x_local = prepare_view(
        local_data
    )

    x_global = prepare_view(
        global_data
    )

    print("Local input:", x_local.shape)
    print("Global input:", x_global.shape)

    assert x_local.shape == (
        SAMPLE_COUNT, 1, LENGTH
    )

    assert x_global.shape == (
        SAMPLE_COUNT, 1, LENGTH
    )

    print("\nLoading pretrained Mantis...")

    network = MantisV1(
        device=DEVICE,
        return_transf_layer=2,
        output_token="combined",
    )

    network = network.from_pretrained(
        CHECKPOINT
    )

    network.eval()

    trainer = MantisTrainer(
        device=DEVICE,
        network=network,
    )

    torch.cuda.reset_peak_memory_stats()

    print("\nExtracting local features...")

    z_local = np.asarray(
        trainer.transform(
            x_local,
            batch_size=4,
        ),
        dtype=np.float32,
    )

    print("Local embeddings:", z_local.shape)

    print("\nExtracting global features...")

    z_global = np.asarray(
        trainer.transform(
            x_global,
            batch_size=4,
        ),
        dtype=np.float32,
    )

    print("Global embeddings:", z_global.shape)

    if z_local.shape != (SAMPLE_COUNT, 512):
        raise RuntimeError(
            "Unexpected local embedding shape"
        )

    if z_global.shape != (SAMPLE_COUNT, 512):
        raise RuntimeError(
            "Unexpected global embedding shape"
        )

    features = np.concatenate(
        [z_local, z_global],
        axis=1,
    )

    if features.shape != (SAMPLE_COUNT, 1024):
        raise RuntimeError(
            "Unexpected combined feature shape"
        )

    if not np.isfinite(features).all():
        raise RuntimeError(
            "Non-finite Mantis features"
        )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        "Combined features:",
        features.shape,
    )

    print(
        "Feature mean:",
        float(features.mean()),
    )

    print(
        "Feature std:",
        float(features.std()),
    )

    print(
        "Mean feature variation across samples:",
        float(
            np.std(
                features,
                axis=0,
            ).mean()
        ),
    )

    memory_mb = (
        torch.cuda.max_memory_allocated()
        / 1024**2
    )

    print(
        "Peak GPU memory:",
        f"{memory_mb:.2f} MB",
    )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        OUTPUT
        / "pilot_embeddings_train.npz"
    )

    np.savez_compressed(
        output_file,
        train_indices=selected,
        z_local=z_local,
        z_global=z_global,
        features=features,
    )

    print("\nSaved:", output_file)

    print("\nTEST SET NOT USED")
    print("VALIDATION SET NOT USED")

    print(
        "\nMANTIS TESS PILOT SUCCESSFUL"
    )


if __name__ == "__main__":
    main()
