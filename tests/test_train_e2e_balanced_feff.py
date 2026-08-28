from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

for dependency in ("lightning", "dgl", "pymatgen", "matgl"):
    pytest.importorskip(dependency)

sys.path.insert(0, str(Path(__file__).parents[1] / "tutorial_omnixas"))
import train_e2e_balanced_feff as training  # noqa: E402


def test_resume_sampler_epoch_uses_checkpoint_epoch(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(training, "torch_load", lambda _path: {"epoch": 7})

    assert training.resume_sampler_epoch(tmp_path / "last.ckpt") == 8


@pytest.mark.parametrize(
    "state",
    [{}, {"epoch": 1.5}, {"epoch": -1}, {"epoch": True}],
)
def test_resume_sampler_epoch_rejects_malformed_epoch(
    monkeypatch,
    tmp_path: Path,
    state,
) -> None:
    monkeypatch.setattr(training, "torch_load", lambda _path: state)

    with pytest.raises(ValueError, match="epoch"):
        training.resume_sampler_epoch(tmp_path / "last.ckpt")


def test_resume_allows_last_checkpoint_without_best(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "last.ckpt").touch()

    training.ensure_clean_or_complete(
        checkpoint_dir,
        "E2E checkpoints",
        allow_last=True,
    )


def test_resume_requires_last_checkpoint(tmp_path: Path) -> None:
    args = SimpleNamespace(resume=True)

    with pytest.raises(FileNotFoundError, match="last checkpoint"):
        training.train_e2e(
            tmp_path,
            tmp_path,
            tmp_path,
            args,
            None,
            None,
            None,
            1,
            1,
        )
