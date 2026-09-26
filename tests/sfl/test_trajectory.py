import numpy as np
import pytest
import torch

from olmo.sfl.trajectory import (
    DeltaTransitionModel,
    SuccessiveClauseDataset,
    TrajectoryArrays,
    load_trajectory_arrays,
    successive_clause_indices,
    trajectory_loss,
)


def test_successive_clause_indices_respect_document_boundaries():
    current, following = successive_clause_indices(
        np.array(["a", "a", "b", "b", "b"]), np.array([2, 1, 3, 1, 2])
    )
    assert list(zip(current, following)) == [(1, 0), (3, 4), (4, 2)]


def test_dataset_and_loss_use_nine_dimensional_states():
    matrices = np.arange(36, dtype=np.float32).reshape(4, 3, 3)
    arrays = TrajectoryArrays(
        matrices=matrices,
        deltas=np.zeros_like(matrices),
        doc_ids=np.array(["a", "a", "b", "b"]),
        clause_ids=np.array([1, 2, 1, 2]),
    )
    dataset = SuccessiveClauseDataset(arrays)
    current, change, next_state = dataset[0]
    assert current.shape == change.shape == next_state.shape == (9,)
    model = DeltaTransitionModel(hidden_dim=12)
    batch = torch.stack([current, current])
    prediction = model(batch)
    assert prediction.shape == (2, 9)
    assert torch.isfinite(trajectory_loss(batch, torch.stack([change, change]), torch.stack([next_state, next_state]), prediction))


def test_loader_rejects_missing_schema_arrays(tmp_path):
    path = tmp_path / "broken.npz"
    np.savez(path, matrices=np.zeros((1, 3, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="missing arrays"):
        load_trajectory_arrays(path)
