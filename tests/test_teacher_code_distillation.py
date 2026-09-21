"""Scientific invariants and tiny CPU end-to-end KD pipeline tests; no downloads."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from src.mech import teacher_code_distillation as kd
from scripts import run_teacher_code_distillation as run

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(1)
    torch.manual_seed(19)


class Unlimited:
    def check(self):
        pass


class InterruptAfter:
    def __init__(self,n):
        self.n=n
    def check(self):
        self.n-=1
        if self.n<0:
            raise kd.BudgetReached("test interruption")


def tiny_model(vocab=128,width=128):
    return GPT2LMHeadModel(GPT2Config(vocab_size=vocab,n_embd=width,n_layer=1,n_head=4,
        n_positions=128,n_ctx=128,bos_token_id=1,eos_token_id=1,pad_token_id=1,
        resid_pdrop=.1,embd_pdrop=.1,attn_pdrop=.1))


def rows(n=4):
    result=[]
    for i in range(n):
        result.append(dict(id=str(i),question=f"question {i}",gold="1",
            ids=[2,3+i,4,5,6,1],prompt_length=2,count=4,offset=4*i,rationale_length=2))
    return result


def test_causal_shift_and_eos():
    row=rows(1)[0]
    labels,positions=kd.labels_and_positions(row)
    assert labels.tolist()==[4,5,6,1]
    assert positions.tolist()==[1,2,3,4]
    assert row["ids"][positions[-1]]==6  # State predicts EOS; does not consume it.


def test_topk_tail_is_full_kd_when_only_one_token_is_omitted():
    logits=torch.randn(4,9,requires_grad=True)
    lp=torch.randn(4,9).log_softmax(-1)
    values,indices=lp.topk(8,-1)
    tail=lp.scatter(1,indices,-torch.inf).logsumexp(-1)
    full=kd.soft_cross_entropy(logits,lp.exp()).mean()
    sparse=kd.sparse_kd(logits,dict(kind="topk",indices=indices,logp=values,tail=tail)).mean()
    assert torch.allclose(full,sparse,atol=1e-6)
    fg=torch.autograd.grad(full,logits,retain_graph=True)[0]
    sg=torch.autograd.grad(sparse,logits)[0]
    assert torch.allclose(fg,sg,atol=1e-6)


def test_sampled_gradient_approximates_dense_gradient():
    logits=torch.randn(1,12,requires_grad=True)
    p=torch.randn(1,12).softmax(-1)
    indices=torch.multinomial(p,200000,replacement=True)
    full=kd.soft_cross_entropy(logits,p).mean()
    sampled=kd.sparse_kd(logits,dict(kind="sample",indices=indices)).mean()
    a=torch.autograd.grad(full,logits,retain_graph=True)[0]
    b=torch.autograd.grad(sampled,logits)[0]
    assert torch.allclose(a,b,atol=.002)


def test_exact_budget_includes_decoder_and_uint16_ids():
    costs=kd.exact_budget(500000,96,50257)
    assert costs["k"]==51 and costs["m"]==105
    assert costs["topk_bytes"]<=costs["lr_bytes"]
    assert costs["sample_bytes"]<=costs["lr_bytes"]
    assert costs["lr_bytes"]==4096+96000000+9749858
    with pytest.raises(ValueError):
        kd.exact_budget(0,96,50257)


def test_paired_statistics_do_not_count_seeds_as_questions():
    a=np.zeros((3,40),bool); b=a.copy(); b[:,:8]=True
    report=kd.paired_summary(a,b,samples=200,seed=3)
    assert report["delta_pp"]==pytest.approx(20)
    assert report["changed_wrong_to_correct"]==[8,8,8]
    assert report["crossed_ci95_pp"][0]>0
    assert kd.holm([.01,.03])==[True,True]
    assert kd.holm([.04,.06])==[False,False]


def test_training_resumes_exactly_and_chunks_have_correct_labels(tmp_path):
    original=tiny_model(vocab=32,width=32)
    cfg=kd.Settings(epochs=2,effective_batch=2,chunk_tokens=2,checkpoint_steps=1)
    data=rows()
    class SFT:
        arm="sft"
        def get(self,start,stop):
            return None
    baseline=copy.deepcopy(original)
    run.train_student(baseline,data,SFT(),tmp_path/"baseline",cfg,89,"cpu",Unlimited())
    interrupted=copy.deepcopy(original)
    with pytest.raises(kd.BudgetReached):
        run.train_student(interrupted,data,SFT(),tmp_path/"resume",cfg,89,"cpu",InterruptAfter(2))
    restarted=copy.deepcopy(original)
    run.train_student(restarted,data,SFT(),tmp_path/"resume",cfg,89,"cpu",Unlimited())
    for name,value in baseline.state_dict().items():
        assert torch.equal(value,restarted.state_dict()[name]),name
    assert not (tmp_path/"resume"/"resume.pt").exists()
    # Real gradient updates happened.
    assert not torch.equal(baseline.transformer.wte.weight,original.transformer.wte.weight)


def test_tiny_cache_codec_packages_kd_and_gradient_audit(tmp_path):
    cfg=kd.Settings(smoke=True,secondary=True).checked()
    teacher=tiny_model()
    data=rows(5)
    for split in ("fit","select","audit","train"):
        run.collect_states(teacher,data,tmp_path/"states"/split,"cpu",Unlimited())
    kd.save_torch(tmp_path/"teacher_readout.pt",dict(weight=teacher.lm_head.weight.detach().half()))
    for name in ("whitened","weight_svd"):
        run.fit_codec(tmp_path,cfg,"cpu",Unlimited(),name)
    packages=run.build_target_packages(tmp_path,cfg,"cpu",Unlimited())
    assert {p["arm"] for p in packages}==set(kd.PRIMARY[1:]+kd.SECONDARY)
    full=run.TargetReader(tmp_path,"full",89,"cpu")
    h=run.hidden_cache(tmp_path,"train")[:3]
    expected=(torch.nn.functional.linear(torch.from_numpy(np.array(h)).float(),
              teacher.lm_head.weight.detach().half().float())/2).softmax(-1)
    assert torch.allclose(full.get(0,3)["probability"],expected)
    for arm in kd.PRIMARY:
        model=tiny_model()
        reader=run.TargetReader(tmp_path,arm,89,"cpu")
        result=run.train_student(model,data,reader,tmp_path/"students"/arm,cfg,89,"cpu",Unlimited())
        assert len(result["history"])==3
        assert np.isfinite(result["history"][-1]["loss"])
    diagnostic=run.gradient_audit(tmp_path,tiny_model(),data,cfg,"cpu",Unlimited(),"tiny")
    assert len(diagnostic)==1
    assert "parameter_combined" in diagnostic[0]["arms"]["lr96"]
    strata=run.target_strata_audit(tmp_path,data,cfg,"cpu",Unlimited())
    assert {r["token_type"] for r in strata}=={"rationale","answer","eos"}


def test_notebook_embeds_current_source_and_has_no_outputs():
    path=ROOT/"notebooks/kaggle_teacher_code_distillation.ipynb"
    notebook=json.loads(path.read_text(encoding="utf-8"))
    import nbformat
    nbformat.validate(nbformat.from_dict(notebook))
    code=["".join(c["source"]) for c in notebook["cells"] if c["cell_type"]=="code"]
    for i,source in enumerate(code):
        compile(source,f"cell_{i}","exec")
    assert all(not c["outputs"] and c["execution_count"] is None
               for c in notebook["cells"] if c["cell_type"]=="code")
    # Evaluate only the generated literal definitions; no network/setup execution.
    ns={}
    exec(code[1],ns)
    for relative,source in ns["EMBEDDED_FILES"].items():
        assert source==(ROOT/relative).read_text(encoding="utf-8-sig")
    assert "STAGE = \"pilot\"" in code[0]
    assert "run_experiment" in "\n".join(code)





def test_full_smoke_orchestration_and_filtered_resume(tmp_path,monkeypatch):
    """Run the production pilot/full orchestration on CPU with no network/model downloads."""
    class Tokens:
        eos_token_id=1
        def encode(self,text,add_special_tokens=False):
            return [2,3]
        def decode(self,ids,skip_special_tokens=True):
            return "1"
        def convert_ids_to_tokens(self,index):
            return str(index)
    tok=Tokens()
    monkeypatch.setattr(kd,"VOCAB",128)
    monkeypatch.setattr(run,"student_tokenizer",lambda:tok)
    monkeypatch.setattr(run,"load_student",lambda device:tiny_model())
    monkeypatch.setattr(run,"load_teacher",lambda device:(tiny_model().eval(),tok,{"test":True}))
    monkeypatch.setattr(run,"version",lambda package:"test")
    data=rows(4)
    parts={s:copy.deepcopy(data) for s in ("fit","select","audit","dev","train")}
    tests={"gsm8k":copy.deepcopy(data),"svamp":copy.deepcopy(data)}
    monkeypatch.setattr(run,"prepare_data",lambda *args:(parts,tests))
    cfg=kd.Settings(smoke=True,max_length=128)
    root,result=run.run_experiment(tmp_path,cfg,stage="full",source_hash="test",
                                  arm_filter=["sft"],_device="cpu")
    assert result["status"]=="partial"
    assert (root/"analysis_lock.json").exists()
    assert not (root/"completion.json").exists()
    _,result=run.run_experiment(tmp_path,kd.Settings(smoke=True,max_length=128),stage="full",
                               source_hash="test",_device="cpu")
    assert (root/"completion.json").exists()
    assert result["gates"]["eligible_for_claims"] is False
    assert set(result["datasets"]["gsm8k"]["arms"])==set(kd.PRIMARY)


