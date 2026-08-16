from __future__ import annotations

import pandas as pd

from research_audit_v2.demographic_composition.cohorts import build_scenarios


COUNTS = {
    "White": 18_612,
    "Latino_Hispanic": 14_990,
    "East Asian": 13_837,
    "Indian": 13_835,
    "Black": 13_789,
    "Southeast Asian": 12_210,
    "Middle Eastern": 10_425,
}


def _catalog() -> pd.DataFrame:
    rows = []
    for group, count in COUNTS.items():
        rows.extend(
            {
                "relative_path": f"{group}/{index}.jpg",
                "source_race_label": group,
            }
            for index in range(count)
        )
    return pd.DataFrame(rows)


def _config() -> dict[str, object]:
    return {
        "random_seed": 20260815,
        "sample_size": 36_456,
        "group_column": "source_race_label",
        "perturbed_group": "Middle Eastern",
    }


def test_declared_scenarios_have_exact_quotas_without_replacement():
    selected, _ = build_scenarios(_catalog(), _config())
    observed = selected.groupby(["scenario", "source_race_label"]).size().unstack(fill_value=0)

    assert observed.loc["A"].to_dict() == {
        "Black": 5_145,
        "East Asian": 5_163,
        "Indian": 5_163,
        "Latino_Hispanic": 5_594,
        "Middle Eastern": 3_890,
        "Southeast Asian": 4_556,
        "White": 6_945,
    }
    assert observed.loc["B"].eq(5_208).all()
    assert observed.loc["C"].to_dict() == {
        "Black": 5_642,
        "East Asian": 5_642,
        "Indian": 5_642,
        "Latino_Hispanic": 5_642,
        "Middle Eastern": 2_604,
        "Southeast Asian": 5_642,
        "White": 5_642,
    }
    assert observed.loc["D"].to_dict() == {
        "Black": 4_340,
        "East Asian": 4_340,
        "Indian": 4_340,
        "Latino_Hispanic": 4_340,
        "Middle Eastern": 10_416,
        "Southeast Asian": 4_340,
        "White": 4_340,
    }
    assert selected.groupby("scenario").size().eq(36_456).all()
    assert selected.groupby("scenario")["record_id"].nunique().eq(36_456).all()


def test_selection_is_deterministic_and_reserves_follow_the_shared_group_order():
    first, first_reserves = build_scenarios(_catalog(), _config())
    second, second_reserves = build_scenarios(_catalog(), _config())

    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first_reserves, second_reserves)
    selected_union = set(first["record_id"])
    assert selected_union.isdisjoint(first_reserves["record_id"])
    assert first_reserves.groupby("source_race_label")["selection_rank"].min().to_dict() == {
        "Black": 5_642,
        "East Asian": 5_642,
        "Indian": 5_642,
        "Latino_Hispanic": 5_642,
        "Middle Eastern": 10_416,
        "Southeast Asian": 5_642,
        "White": 6_945,
    }


def test_catalog_contract_rejects_missing_groups_and_duplicate_paths():
    catalog = _catalog()
    duplicate = pd.concat([catalog, catalog.iloc[[0]]], ignore_index=True)

    for invalid in (catalog[catalog["source_race_label"] != "Indian"], duplicate):
        try:
            build_scenarios(invalid, _config())
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid FairFace catalog was accepted")
