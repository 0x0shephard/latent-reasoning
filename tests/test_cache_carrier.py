import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.data.official_codi_training import collate_official_codi_kv_rows  # noqa: E402
from src.mech.cache_carrier import (  # noqa: E402
    CacheEdit,
    donor_view,
    generate_under,
    latent_path,
    outcomes_under,
    record_native_trajectory,
    record_trajectory,
)
from tests.test_trajectory_supervision import ROWS, CharTokenizer, _tiny_model  # noqa: E402

CPU = torch.device("cpu")


class _Batch(dict):
    def to(self, device):
        return _Batch({k: v.to(device) for k, v in self.items()})


class BatchCharTokenizer(CharTokenizer):
    """The character tokenizer with the batched, left-padded call the released
    generation path makes; unknown characters map to id 1."""

    def encode(self, text):
        return [self.vocab.get(c, 1) for c in text]

    def __call__(self, text, **kwargs):
        if isinstance(text, str):
            return {"input_ids": self.encode(text)}
        ids = [self.encode(t) for t in text]
        width = max(len(i) for i in ids)
        padded = [[self.pad_token_id] * (width - len(i)) + i for i in ids]
        mask = [[0] * (width - len(i)) + [1] * len(i) for i in ids]
        return _Batch({"input_ids": torch.tensor(padded), "attention_mask": torch.tensor(mask)})

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(self.inverse.get(int(i), "") for i in ids)


def _setup():
    model, tokenizer = _tiny_model().eval(), CharTokenizer()
    traj, pred, gold = record_trajectory(model, tokenizer, ROWS, latent_positions=6, batch_size=2, device=CPU)
    return model, tokenizer, traj, pred, gold


def test_recording_shapes_and_gold():
    model, tokenizer, traj, pred, gold = _setup()
    n, layers, heads, hd = len(ROWS), 2, 2, 8
    assert traj.states.shape == (n, 6, 16)
    assert traj.keys.shape == traj.values.shape == (n, 6, layers, heads, hd)
    batch = collate_official_codi_kv_rows(tokenizer, ROWS, bot_token_id=model.bot_id)
    assert torch.equal(gold, batch.teacher_ids[torch.arange(n), batch.teacher_answer_start])


def test_self_donor_is_an_identity_on_both_routes():
    """Patching a row's own recorded state or K/V must reproduce the baseline exactly;
    this pins the slot-to-cache-position convention."""
    model, tokenizer, traj, pred, gold = _setup()
    for slot in range(6):
        own_hidden = outcomes_under(model, tokenizer, ROWS, latent_positions=6, batch_size=2, device=CPU,
                                    hidden_donors={slot: traj.states[:, slot]})
        own_kv = outcomes_under(model, tokenizer, ROWS, latent_positions=6, batch_size=2, device=CPU,
                                cache_edits=[CacheEdit(slot, (0, 1), True, True)], donor=traj)
        assert torch.equal(own_hidden, pred) and torch.equal(own_kv, pred)
    # exact logits too, so the check is not only on the argmax; the donor must be recorded
    # at the same batch width, because GPT-2's left padding sets the absolute position ids
    full, _, _ = record_trajectory(model, tokenizer, ROWS, latent_positions=6, batch_size=len(ROWS), device=CPU)
    batch = collate_official_codi_kv_rows(tokenizer, ROWS, bot_token_id=model.bot_id)
    with torch.no_grad():
        base, _, _ = latent_path(model, batch, latent_positions=6)
        same, _, _ = latent_path(model, batch, latent_positions=6, cache_edits=[CacheEdit(2, (0, 1), True, True)],
                                 donor=full, rows=slice(0, len(ROWS)), hidden_donors={4: full.states[:, 4]})
    assert torch.allclose(base, same, atol=1e-6)


def test_foreign_donor_changes_the_computation_through_each_route():
    model, tokenizer, traj, pred, gold = _setup()
    donor = donor_view(traj, [1, 2, 0])
    batch = collate_official_codi_kv_rows(tokenizer, ROWS, bot_token_id=model.bot_id)
    with torch.no_grad():
        base, _, _ = latent_path(model, batch, latent_positions=6)
        via_hidden, _, _ = latent_path(model, batch, latent_positions=6, hidden_donors={1: donor.states[:, 1]})
        via_v, _, _ = latent_path(model, batch, latent_positions=6, cache_edits=[CacheEdit(1, (0, 1), False, True)],
                                  donor=donor, rows=slice(0, 3))
        via_k_layer0, _, _ = latent_path(model, batch, latent_positions=6, cache_edits=[CacheEdit(1, (0,), True, False)],
                                         donor=donor, rows=slice(0, 3))
        terminal_hidden, _, _ = latent_path(model, batch, latent_positions=6, hidden_donors={5: donor.states[:, 5]})
        terminal_kv, _, _ = latent_path(model, batch, latent_positions=6, cache_edits=[CacheEdit(5, (0, 1), True, True)],
                                        donor=donor, rows=slice(0, 3))
    assert not torch.allclose(base, via_hidden, atol=1e-6)
    assert not torch.allclose(base, via_v, atol=1e-6)
    assert not torch.allclose(base, via_k_layer0, atol=1e-6)
    # the terminal thought's output state is discarded by the released path, its K/V are not
    assert torch.allclose(base, terminal_hidden, atol=1e-6)
    assert not torch.allclose(base, terminal_kv, atol=1e-6)


def test_native_path_recording_and_self_donor_identity():
    model, tokenizer = _tiny_model().eval(), BatchCharTokenizer()
    questions = [r["question"] for r in ROWS]
    traj, outputs = record_native_trajectory(model, tokenizer, questions, latent_iterations=6, batch_size=2, device=CPU,
                                             max_new_tokens=4)
    assert traj.states.shape == (3, 6, 16) and traj.keys.shape == (3, 6, 2, 2, 8) and len(outputs) == 3
    same = generate_under(model, tokenizer, questions, latent_iterations=6, batch_size=2, device=CPU, max_new_tokens=4,
                          hidden_donors={1: traj.states[:, 1]}, cache_edits=[CacheEdit(3, (0, 1), True, True)], donor=traj)
    assert same == outputs
    swapped = generate_under(model, tokenizer, questions, latent_iterations=6, batch_size=2, device=CPU, max_new_tokens=4,
                             cache_edits=[CacheEdit(s, (0, 1), True, True) for s in range(6)], donor=donor_view(traj, [1, 2, 0]))
    assert len(swapped) == 3
