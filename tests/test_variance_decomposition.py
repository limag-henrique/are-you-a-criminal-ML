import numpy as np
import pandas as pd

from face_profile_ml.variance_decomposition import decompose_variance


def test_variance_decomposition_identifies_dominant_fixed_effect() -> None:
    rows = []
    rng = np.random.default_rng(8)
    for scenario in ["A", "B"]:
        for replication in range(8):
            for seed in range(3):
                rows.append(
                    {
                        "scenario": scenario,
                        "replication": replication,
                        "seed": seed,
                        "k": 4,
                        "auc": (0.2 if scenario == "A" else 0.9) + rng.normal(0, 0.01),
                    }
                )

    result = decompose_variance(
        pd.DataFrame(rows), "auc", ["scenario", "replication", "seed", "k"]
    )

    fractions = result.as_frame().set_index("factor")["fraction_explained"]
    assert fractions["scenario"] > 0.9
    assert np.isclose(fractions.sum(), 1.0)

