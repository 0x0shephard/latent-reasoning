"""The CODI wrapper generalised to the LLaMA family (ledger §114)."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.data.official_codi_training import encode_official_codi_row  # noqa: E402
from src.mech.cache_carrier import backbone_layers, group_names, layer_groups  # noqa: E402
from src.models.official_codi import (  # noqa: E402
    GPT2_LORA_TARGET_MODULES,
    LLAMA_LORA_TARGET_MODULES,
    OfficialCODIGPT2,
    prepare_backbone_config,
)
from tests.test_trajectory_supervision import ROWS, CharTokenizer, _tiny_model  # noqa: E402


def test_layer_groups_reproduce_gpt2_names_and_scale_to_llama_depth():
    assert layer_groups(12) == {"0_7": tuple(range(8)), "8_9": (8, 9), "10_11": (10, 11)}
    assert group_names(12) == {"early": "0_7", "mid": "8_9", "late": "10_11"}
    g = layer_groups(16)
    assert list(g) == ["0_10", "11_12", "13_15"] and g["0_10"] == tuple(range(11)) and g["13_15"] == (13, 14, 15)
    g = layer_groups(28)
    assert sum(len(v) for v in g.values()) == 28 and list(g)[0].startswith("0_")
    with pytest.raises(ValueError):
        layer_groups(4)


def test_lora_target_constants_match_the_released_test_script():
    assert GPT2_LORA_TARGET_MODULES == ("c_attn", "c_proj", "c_fc")
    assert LLAMA_LORA_TARGET_MODULES == ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj")


def test_tiny_gpt2_wrapper_unchanged_and_depth_readable():
    model = _tiny_model()
    assert model.question_special_tokens is False
    assert backbone_layers(model) == 2
    assert model.input_embeddings() is model.codi.get_base_model().transformer.wte


def test_tiny_llama_wrapper_builds_with_llama_targets_and_generic_embeddings():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=60, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
                         num_key_value_heads=1, max_position_embeddings=128)
    model = OfficialCODIGPT2(LlamaForCausalLM(prepare_backbone_config(config)), lora_rank=4, lora_alpha=8, lora_dropout=0.0, projection_dim=16,
                             lora_target_modules=LLAMA_LORA_TARGET_MODULES, question_special_tokens=True)
    assert (model.pad_token_id, model.bot_id, model.eot_id) == (60, 61, 62)
    assert model.input_embeddings().weight.shape[0] == 63
    assert backbone_layers(model) == 2 and model.question_special_tokens is True
    names = {n for n, _ in model.named_parameters() if "lora_A" in n}
    assert any("q_proj" in n for n in names) and any("gate_proj" in n for n in names)
    state = model.state_dict()
    assert any("embed_tokens.weight" in k for k in state) and not any("wte.weight" in k for k in state)


def test_answer_space_token_shift_only_when_flagged():
    tokenizer = CharTokenizer()
    plain = encode_official_codi_row(tokenizer, ROWS[0], bot_token_id=61)
    flagged = CharTokenizer()
    flagged.official_codi_answer_space_token = True
    shifted = encode_official_codi_row(flagged, ROWS[0], bot_token_id=61)
    # the character tokenizer emits the space after "The answer is:" as its own token
    assert tokenizer.decode([plain.teacher_ids[plain.teacher_answer_start]]) == " "
    assert shifted.teacher_answer_start == plain.teacher_answer_start + 1
    assert shifted.teacher_endpoint == plain.teacher_endpoint + 1
    assert tokenizer.decode([shifted.teacher_ids[shifted.teacher_answer_start]]) == "5"
    assert shifted.teacher_ids == plain.teacher_ids and shifted.student_question_ids == plain.student_question_ids


def _tiny_llama(pad_aware, lean_logits=False):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=60, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
                         num_key_value_heads=1, max_position_embeddings=128)
    model = OfficialCODIGPT2(LlamaForCausalLM(prepare_backbone_config(config)), lora_rank=4, lora_alpha=8, lora_dropout=0.0,
                             projection_dim=16, lora_target_modules=LLAMA_LORA_TARGET_MODULES, question_special_tokens=False,
                             pad_aware_generation=pad_aware, prompt_logits_last_only=lean_logits)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0, 0.1)
    return model.eval()


def test_pad_aware_path_makes_left_padding_invisible_on_llama():
    from src.data.official_codi_training import collate_official_codi_kv_rows
    from src.mech.cache_carrier import latent_path

    tokenizer = CharTokenizer()
    rows = ROWS  # three questions of different lengths -> left padding in one batch
    for pad_aware, expect_equal in ((True, True), (False, False)):
        model = _tiny_llama(pad_aware)
        batch = collate_official_codi_kv_rows(tokenizer, rows, bot_token_id=model.bot_id)
        with torch.no_grad():
            batched, _, _ = latent_path(model, batch, latent_positions=6)
            singles = []
            for row in rows:
                one = collate_official_codi_kv_rows(tokenizer, [row], bot_token_id=model.bot_id)
                logits, _, _ = latent_path(model, one, latent_positions=6)
                singles.append(logits[0])
        same = torch.allclose(batched, torch.stack(singles), atol=1e-4)
        assert same == expect_equal, f"pad_aware={pad_aware}: padded batch equals unpadded rows? {same}"


def test_accuracy_gate_direction():
    from src.eval.official_codi_gate import build_accuracy_gate

    common = dict(evaluated_counts={"gsm8k": 1319}, expected_counts={"gsm8k": 1319}, published_accuracy={"gsm8k": 0.519},
                  primary_dataset="gsm8k", absolute_tolerance=0.03)
    assert build_accuracy_gate(results={"gsm8k": 0.555}, **common)["status"] == "failed"
    assert build_accuracy_gate(results={"gsm8k": 0.555}, direction="at_least", **common)["status"] == "passed"
    assert build_accuracy_gate(results={"gsm8k": 0.48}, direction="at_least", **common)["status"] == "failed"
    assert build_accuracy_gate(results={"gsm8k": 0.50}, direction="at_least", **common)["status"] == "passed"
    with pytest.raises(ValueError):
        build_accuracy_gate(results={"gsm8k": 0.5}, direction="sideways", **common)


def test_prompt_logits_last_only_leaves_hidden_states_and_generation_unchanged():
    from src.mech.cache_carrier import record_trajectory
    from src.models.official_codi import generate_official_codi, prompt_logits_kwargs
    from tests.test_cache_carrier import BatchCharTokenizer

    tokenizer = BatchCharTokenizer()
    full, lean = _tiny_llama(True, lean_logits=False), _tiny_llama(True, lean_logits=True)
    lean.load_state_dict(full.state_dict())
    assert prompt_logits_kwargs(full) == {} and prompt_logits_kwargs(lean) == {"logits_to_keep": 1}
    a, pa, _ = record_trajectory(full, tokenizer, ROWS, latent_positions=6, batch_size=2, device=torch.device("cpu"))
    b, pb, _ = record_trajectory(lean, tokenizer, ROWS, latent_positions=6, batch_size=2, device=torch.device("cpu"))
    assert torch.allclose(a.states, b.states, atol=1e-6) and torch.equal(pa, pb)
    questions = [r["question"] for r in ROWS]
    with torch.inference_mode():
        ga = generate_official_codi(full, tokenizer, questions, latent_iterations=6, max_new_tokens=4, batch_size=2, device=torch.device("cpu"))
        gb = generate_official_codi(lean, tokenizer, questions, latent_iterations=6, max_new_tokens=4, batch_size=2, device=torch.device("cpu"))
    assert ga == gb
