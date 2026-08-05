import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("lightning")


SPEC = spec_from_file_location(
    "train_anionxas_universal",
    Path(__file__).parents[1] / "tutorial_omnixas" / "train_anionxas_universal.py",
)
assert SPEC and SPEC.loader
training = module_from_spec(SPEC)
sys.modules[SPEC.name] = training
SPEC.loader.exec_module(training)


def test_balanced_relative_mse_uses_macro_element_medians() -> None:
    value = training.balanced_relative_mse(
        torch.tensor([1.0, 3.0, 4.0, 8.0]),
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([2.0, 4.0]),
    )
    assert value.item() == pytest.approx(1.25)


def test_balanced_module_state_loads_into_plain_module() -> None:
    balanced = training.BalancedRelativeMSEModule(
        training.XASBlock(4, [3], 2), np.array([1.0, 2.0]), np.array([3.0, 4.0])
    )
    state = balanced.state_dict()
    assert "training_baselines" not in state
    assert "validation_baselines" not in state

    plain = training.PlModule(training.XASBlock(4, [3], 2))
    plain.load_state_dict(state, strict=True)


def test_baselines_use_each_element_training_mean() -> None:
    data = training.Dataset(
        X=np.zeros((8, 1), dtype=np.float32),
        y=np.array(
            [
                [0, 0],
                [2, 2],
                [0, 0],
                [3, 3],
                [0, 0],
                [2, 2],
                [0, 0],
                [3, 3],
            ],
            dtype=np.float32,
        ),
        elements=np.array(["Cu", "Cu", "Cu", "Cu", "Fe", "Fe", "Fe", "Fe"]),
        material_ids=np.array(
            [
                "cu-train-a",
                "cu-train-b",
                "cu-val",
                "cu-test",
                "fe-train-a",
                "fe-train-b",
                "fe-val",
                "fe-test",
            ]
        ),
        sites=np.arange(8, dtype=np.int64),
        split_codes=np.array([0, 0, 1, 2, 0, 0, 1, 2], dtype=np.int8),
        metadata={},
        names=["Cu", "Fe"],
    )
    train, validation = training.baselines(data, ["Cu", "Fe"])
    assert train.tolist() == pytest.approx([1.0, 1.0])
    assert validation.tolist() == pytest.approx([1.0, 1.0])


def test_head_paths_include_scope_and_objective() -> None:
    hparams = {"name": "test_setting"}
    all_path = training.head_run_dir(
        Path("run"), "universalXAS", "All_elements", training.setting_name(hparams)
    )
    eight_path = training.head_run_dir(
        Path("run"), "universalXAS", "OmniXAS_8", training.setting_name(hparams)
    )
    tuned_path = training.head_run_dir(
        Path("run"),
        "tunedUniversalXAS",
        "OmniXAS_8",
        training.setting_name(hparams),
        "Cu",
    )
    assert all_path != eight_path
    assert all_path.parts[-4:] == (
        "universalXAS",
        "All_elements",
        "runs",
        "test_setting_balanced_relative_mse",
    )
    assert "OmniXAS_8" in tuned_path.parts
    assert "balanced_relative_mse" in tuned_path.name


def test_tuned_manifest_rejects_changed_universal(tmp_path: Path) -> None:
    universal = tmp_path / "universal.ckpt"
    setting = tmp_path / "tuned"
    universal.write_bytes(b"first")
    setting.mkdir()
    (setting / "source_universal.json").write_text(
        json.dumps(training.source_manifest(universal)), encoding="utf-8"
    )
    training.check_manifest(setting, universal)

    universal.write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not match"):
        training.check_manifest(setting, universal)
