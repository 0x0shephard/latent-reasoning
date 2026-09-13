from types import SimpleNamespace

import torch
from torch import nn

from src.mech.official_codi_layerwise import (
    OfficialCODILayerwiseEndpointCollector,
    OfficialCODILayerwiseEndpointIntervention,
    attention_location_names,
    gpt2_qkv_response_bases,
    layerwise_location_names,
    residual_location_names,
)


class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = nn.LayerNorm(768)
        self.attn = SimpleNamespace(c_attn=nn.Linear(768, 2304))

    def forward(self, hidden):
        normalized = self.ln_1(hidden)
        self.attn.c_attn(normalized)
        return hidden + 0.01


class FakeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop = nn.Identity()
        self.h = nn.ModuleList([FakeBlock() for _ in range(12)])
        self.ln_f = nn.LayerNorm(768)

    def forward(self, hidden):
        hidden = self.drop(hidden)
        for block in self.h:
            hidden = block(hidden)
        return self.ln_f(hidden)


class FakeModel:
    def __init__(self):
        self.codi = SimpleNamespace(transformer=FakeTransformer())


def test_location_inventory_is_complete():
    assert len(residual_location_names()) == 14
    assert len(attention_location_names()) == 12
    assert len(layerwise_location_names()) == 62


def test_collector_and_intervention_touch_only_selected_last_rows():
    model = FakeModel()
    transformer = model.codi.transformer
    hidden = torch.randn(3, 4, 768)
    mask = torch.tensor([True, False, True])
    collector = OfficialCODILayerwiseEndpointCollector(
        model, locations=("resid_embed", "attn_ln_00", "query_00", "resid_ln_f")
    )
    with collector.activate(mask):
        transformer(hidden)
    values = collector.stacked(2)
    assert all(value.shape == (2, 768) for value in values.values())

    basis = torch.eye(768)[:, :3]
    intervention = OfficialCODILayerwiseEndpointIntervention(
        model,
        location="resid_embed",
        basis=basis,
        centre=torch.zeros(768),
        mode="remove",
    )
    with intervention.activate(mask):
        edited = transformer.drop(hidden)
    assert torch.equal(edited[1], hidden[1])
    assert torch.equal(edited[:, :-1], hidden[:, :-1])
    assert torch.allclose(edited[mask, -1, :3], torch.zeros(2, 3))


def test_qkv_response_basis_uses_the_effective_affine_module():
    model = FakeModel()
    basis = torch.eye(768)[:, :2]
    values = gpt2_qkv_response_bases(model, {0: basis})[0]
    assert set(values) == {"query", "key", "value"}
    assert all(value.shape == (768, 2) for value in values.values())
