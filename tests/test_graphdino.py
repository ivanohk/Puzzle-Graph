"""
Smoke tests for the GraphDINO pipeline.

These tests verify that:
1. GraphDINO can be instantiated from a config dict.
2. forward() returns (embeddings, loss) with correct shapes.
3. update_ema_params moves teacher weights toward student weights.
4. DINOTrainer.train_epoch() runs without errors and produces finite loss.
"""

import sys, os
import pytest
import torch
from torch_geometric.data import Data, Batch

# ── Make the repo root importable ────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Force module registration by importing the files that call @register
import src.models.encoder.gin          # registers "gin" in ENCODERS
import src.models.head.dino_head       # registers "dino" in HEADS
import src.data.dino_augmentations     # registers augmentations


# ── Helpers ──────────────────────────────────────────────────────────

def _make_config():
    """Minimal config dict mirroring graphdino.yaml."""
    return {
        "encoder": {
            "name": "gin",
            "hidden_dim": 32,
            "num_layers": 2,
        },
        "head": {
            "name": "dino",
            "proj_hidden": 64,
            "bottleneck_dim": 32,
            "n_prototypes": 128,
            "student_temp": 0.1,
            "teacher_temp": 0.04,
            "center_momentum": 0.9,
        },
        "augment": [
            {"name": "edge_drop", "p": 0.3},
            {"name": "feat_mask", "p": 0.2},
        ],
        "ema_tau": 0.996,
    }


def _make_batch(n_graphs=4, n_nodes_per_graph=10, n_features=7):
    """Create a synthetic PyG Batch."""
    graphs = []
    for _ in range(n_graphs):
        n = n_nodes_per_graph
        # Random Erdős–Rényi-ish edges
        src = torch.randint(0, n, (n * 2,))
        dst = torch.randint(0, n, (n * 2,))
        edge_index = torch.stack([src, dst])
        x = torch.randn(n, n_features)
        graphs.append(Data(x=x, edge_index=edge_index))
    return Batch.from_data_list(graphs)


# ── Tests ────────────────────────────────────────────────────────────

class TestGraphDINO:
    """Tests for the GraphDINO model."""

    def setup_method(self):
        # Import here so registration side-effects have already happened
        # Use importlib to handle the hyphen in 'model-types'
        import importlib
        spec = importlib.util.spec_from_file_location(
            "graphdino",
            os.path.join(ROOT, "src", "models", "model-types", "graphdino.py"),
        )
        mod = importlib.util.module_from_spec(spec)

        # Temporarily make 'base' importable for the relative import
        base_spec = importlib.util.spec_from_file_location(
            "base",
            os.path.join(ROOT, "src", "models", "model-types", "base.py"),
        )
        base_mod = importlib.util.module_from_spec(base_spec)
        base_spec.loader.exec_module(base_mod)

        # Create a fake package so relative imports work
        import types
        pkg = types.ModuleType("src.models.model-types")
        pkg.__path__ = [os.path.join(ROOT, "src", "models", "model-types")]
        sys.modules["src.models.model-types"] = pkg
        sys.modules["src.models.model-types.base"] = base_mod

        # Now we can set the package attribute and load the module
        mod.__package__ = "src.models.model-types"
        spec.loader.exec_module(mod)

        self.GraphDINO = mod.GraphDINO
        self.config = _make_config()
        self.in_channels = 7

    def test_instantiation(self):
        """Model builds without error from a config dict."""
        model = self.GraphDINO(self.config, in_channels=self.in_channels)
        assert model.student_enc is not None
        assert model.teacher_enc is not None
        assert model.student_head is not None
        assert model.teacher_head is not None

    def test_teacher_is_frozen(self):
        """All teacher parameters have requires_grad=False."""
        model = self.GraphDINO(self.config, in_channels=self.in_channels)
        for p in model.teacher_enc.parameters():
            assert not p.requires_grad
        for p in model.teacher_head.parameters():
            assert not p.requires_grad

    def test_forward_shapes(self):
        """forward() returns (embeddings, loss) with correct shapes."""
        model = self.GraphDINO(self.config, in_channels=self.in_channels)
        batch = _make_batch(n_features=self.in_channels)
        emb, loss = model(batch)

        # Embeddings: one per graph, dimension = hidden_dim
        assert emb.shape == (4, self.config["encoder"]["hidden_dim"])
        # Loss: scalar
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_last_teacher_out_stored(self):
        """forward() stashes teacher output for centering."""
        model = self.GraphDINO(self.config, in_channels=self.in_channels)
        batch = _make_batch(n_features=self.in_channels)
        model(batch)
        assert model.last_teacher_out is not None
        # Should be 2× batch_size (view1 + view2)
        assert model.last_teacher_out.shape[0] == 2 * 4


class TestEMAUpdate:
    """Tests for the parameter-level EMA utility."""

    def test_teacher_moves_toward_student(self):
        from src.utils.ema import update_ema_params

        student = torch.nn.Linear(4, 4)
        teacher = torch.nn.Linear(4, 4)
        # Set teacher weights to zero
        with torch.no_grad():
            teacher.weight.fill_(0.0)
            teacher.bias.fill_(0.0)

        tau = 0.5
        update_ema_params(student, teacher, tau)

        # teacher = 0.5 * 0 + 0.5 * student = 0.5 * student
        expected = student.weight.data * (1.0 - tau)
        assert torch.allclose(teacher.weight.data, expected, atol=1e-6)


class TestDINOTrainer:
    """Integration test: run 1 epoch on tiny random data."""

    def test_train_epoch_runs(self):
        import importlib, types

        # Same import dance for hyphenated directory
        base_spec = importlib.util.spec_from_file_location(
            "base",
            os.path.join(ROOT, "src", "models", "model-types", "base.py"),
        )
        base_mod = importlib.util.module_from_spec(base_spec)
        base_spec.loader.exec_module(base_mod)

        pkg = types.ModuleType("src.models.model-types")
        pkg.__path__ = [os.path.join(ROOT, "src", "models", "model-types")]
        sys.modules["src.models.model-types"] = pkg
        sys.modules["src.models.model-types.base"] = base_mod

        spec = importlib.util.spec_from_file_location(
            "graphdino",
            os.path.join(ROOT, "src", "models", "model-types", "graphdino.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "src.models.model-types"
        spec.loader.exec_module(mod)
        GraphDINO = mod.GraphDINO

        from src.training.trainer import DINOTrainer
        from torch_geometric.loader import DataLoader

        config = _make_config()
        model = GraphDINO(config, in_channels=7)

        # Only student parameters go into the optimiser
        optimizer = torch.optim.Adam(
            list(model.student_enc.parameters())
            + list(model.student_head.parameters()),
            lr=1e-3,
        )

        # Tiny dataset
        graphs = [
            Data(
                x=torch.randn(10, 7),
                edge_index=torch.stack([
                    torch.randint(0, 10, (20,)),
                    torch.randint(0, 10, (20,)),
                ]),
            )
            for _ in range(8)
        ]
        loader = DataLoader(graphs, batch_size=4)

        trainer = DINOTrainer(grad_clip_norm=3.0)
        avg_loss = trainer.train_epoch(
            model, loader, optimizer, ema_tau=config["ema_tau"]
        )

        assert isinstance(avg_loss, float)
        assert not (avg_loss != avg_loss)  # not NaN
