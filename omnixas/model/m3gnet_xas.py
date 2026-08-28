"""M3GNet-style graph-to-XAS model used by the balanced FEFF experiments.

This module owns the complete neural-network architecture. Training policy,
balanced sampling, hyperparameter sweeps, checkpoint selection, and evaluation
remain in the training scripts.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from matgl.config import DEFAULT_ELEMENTS
from matgl.graph.compute import compute_theta_and_phi, create_line_graph
from matgl.layers import (
    MLP as M3GNetMLP,
    ActivationFunction,
    BondExpansion,
    EmbeddingBlock,
    GatedMLP,
    M3GNetBlock,
    SphericalBesselWithHarmonics,
    ThreeBodyInteractions,
)
from matgl.utils.cutoff import polynomial_cutoff


# Default architecture for the balanced FEFF experiments.
FEATURE_DIM = 64
SPECTRUM_DIM = 141
FEATURE_SCALE = 1000.0

HEAD_HIDDEN_DIMS = (500, 500, 550)
DEFAULT_HEAD_DROPOUT = 0.10

ENCODER_BLOCKS = 3
ENCODER_CUTOFF = 4.0
ENCODER_THREEBODY_CUTOFF = 4.0
RADIAL_BASIS_MAX_N = 3
RADIAL_BASIS_MAX_L = 3
DEFAULT_GNN_DROPOUT = 0.10


class XASSpectralHead(nn.Sequential):
    """Map an absorbing-site representation to a nonnegative XAS spectrum.

    Hidden stages are ``Linear -> BatchNorm1d -> SiLU -> Dropout``. The output
    stage is ``Linear -> Softplus``.
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dims: Sequence[int] = HEAD_HIDDEN_DIMS,
        output_dim: int = SPECTRUM_DIM,
        dropout: float = DEFAULT_HEAD_DROPOUT,
    ) -> None:
        layers: list[nn.Module] = []
        input_width = input_dim
        for hidden_width in hidden_dims:
            layers.extend(
                [
                    nn.Linear(input_width, hidden_width),
                    nn.BatchNorm1d(hidden_width),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            input_width = hidden_width

        layers.extend([nn.Linear(input_width, output_dim), nn.Softplus()])
        super().__init__(*layers)


class M3GNetXASEncoder(nn.Module):
    """Encode a graph as the node state at its absorbing site."""

    def __init__(
        self,
        dropout: float = DEFAULT_GNN_DROPOUT,
        feature_dim: int = FEATURE_DIM,
        blocks: int = ENCODER_BLOCKS,
        cutoff: float = ENCODER_CUTOFF,
        threebody_cutoff: float = ENCODER_THREEBODY_CUTOFF,
    ) -> None:
        super().__init__()
        activation = ActivationFunction["swish"].value()
        basis_dim = RADIAL_BASIS_MAX_N * RADIAL_BASIS_MAX_L

        self.element_types = DEFAULT_ELEMENTS
        self.cutoff = cutoff
        self.threebody_cutoff = threebody_cutoff

        self.bond_expansion = BondExpansion(
            RADIAL_BASIS_MAX_N,
            RADIAL_BASIS_MAX_L,
            cutoff,
        )
        self.basis_expansion = SphericalBesselWithHarmonics(
            RADIAL_BASIS_MAX_N,
            RADIAL_BASIS_MAX_L,
            cutoff,
            use_smooth=False,
            use_phi=False,
        )
        self.embedding = EmbeddingBlock(
            degree_rbf=basis_dim,
            dim_node_embedding=feature_dim,
            dim_edge_embedding=feature_dim,
            ntypes_node=len(self.element_types),
            activation=activation,
        )
        self.three_body_interactions = nn.ModuleList(
            [
                ThreeBodyInteractions(
                    update_network_atom=M3GNetMLP(
                        dims=[feature_dim, basis_dim],
                        activation=nn.Sigmoid(),
                        activate_last=True,
                    ),
                    update_network_bond=GatedMLP(
                        in_feats=basis_dim,
                        dims=[feature_dim],
                        use_bias=False,
                    ),
                )
                for _ in range(blocks)
            ]
        )
        self.graph_layers = nn.ModuleList(
            [
                M3GNetBlock(
                    degree=basis_dim,
                    activation=activation,
                    conv_hiddens=[feature_dim, feature_dim],
                    dim_node_feats=feature_dim,
                    dim_edge_feats=feature_dim,
                    dropout=dropout,
                )
                for _ in range(blocks)
            ]
        )

    def forward(self, graph, site: torch.Tensor) -> torch.Tensor:
        bond_distances = graph.edata["bond_dist"]
        radial_basis = self.bond_expansion(bond_distances)
        graph.edata["rbf"] = radial_basis

        # MatGL 0.8.5 only constructs line graphs on the CPU. Move the result
        # back so the remaining operations stay on the model's device.
        line_graph = create_line_graph(graph.to("cpu"), self.threebody_cutoff)
        line_graph = line_graph.to(graph.device)
        line_graph.apply_edges(compute_theta_and_phi)

        three_body_basis = self.basis_expansion(line_graph)
        three_body_cutoff = polynomial_cutoff(
            bond_distances, self.threebody_cutoff
        )
        node_features, edge_features, state_features = self.embedding(
            graph.ndata["node_type"], radial_basis, None
        )

        for interaction, graph_layer in zip(
            self.three_body_interactions,
            self.graph_layers,
            strict=True,
        ):
            edge_features = interaction(
                graph,
                line_graph,
                three_body_basis,
                three_body_cutoff,
                node_features,
                edge_features,
            )
            edge_features, node_features, state_features = graph_layer(
                graph,
                edge_features,
                node_features,
                state_features,
            )

        return node_features[site]


class M3GNetXAS(nn.Module):
    """Complete graph-to-spectrum model from the balanced FEFF experiments."""

    def __init__(
        self,
        gnn_dropout: float = DEFAULT_GNN_DROPOUT,
        head_dropout: float = DEFAULT_HEAD_DROPOUT,
        feature_scale: float = FEATURE_SCALE,
    ) -> None:
        super().__init__()
        self.encoder = M3GNetXASEncoder(dropout=gnn_dropout)
        self.head = XASSpectralHead(dropout=head_dropout)
        self.feature_scale = feature_scale

    def encode(
        self,
        graph,
        site: torch.Tensor,
        *,
        scaled: bool = False,
    ) -> torch.Tensor:
        """Return the absorbing-site features, optionally in head-training scale."""
        features = self.encoder(graph, site)
        return features * self.feature_scale if scaled else features

    def forward(self, graph, site: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(graph, site, scaled=True))
