from dataclasses import replace
import math
import json
import pytest
import torch
from sa_jev.smoke import fixture_questions
from sa_jev.schema import Evidence, Pool, State, labels_for, TaskExample
from sa_jev.data import task_states, training_pool, split_questions, audit_splits
from sa_jev.prompts import requests, View
from sa_jev.model import tiny_model, Prediction
from sa_jev.losses import task_loss, stability_loss, dpo_loss
from sa_jev.trajectories import Trajectory, trajectory_log_probability, build_preferences
from sa_jev.retrieval import PoolCache, require_retrieved
from sa_jev.metrics import paired_bootstrap, js_divergence


@pytest.fixture
def sample():
    q = fixture_questions()[0]
    pool = Pool(q.id, q.evidence, "fixture")
    return q, State(q.id, q.group_id, q.question, pool, budget=3)


def test_marginal_relevance_sufficiency_are_distinct(sample):
    q, s = sample
    y = labels_for(q, s)
    assert y.rank == (.5, .5, 0., 0., 0., 0.)
    s = s.read(q.evidence[0].id); y = labels_for(q, s)
    assert y.rank[0] == 1 and y.sufficient == 0
    s = s.read(q.evidence[1].id); y = labels_for(q, s)
    assert y.sufficient == 1 and not y.rank_valid
    duplicate_support = Evidence.make("Another source", q.evidence[0].text + " Verified duplicate support.")
    q = replace(q, coverage={**q.coverage, duplicate_support.id: ("u0",)})
    pool = replace(s.pool, candidates=s.pool.candidates + (duplicate_support,))
    y = labels_for(q, replace(s, pool=pool))
    assert y.relevant[-1] == 1 and y.gains[-1] == 0


def test_unknown_does_not_become_negative(sample):
    q, s = sample
    unknown = Evidence.make("Unknown alternative", "New material with unverified relevance.")
    s = replace(s, pool=replace(s.pool, candidates=s.pool.candidates+(unknown,)))
    y = labels_for(q, s)
    assert not y.rank_valid and not y.rel_valid[-1]
    assert not labels_for(q, s.read(unknown.id)).suf_valid
    assert not labels_for(replace(q, gold_complete=False), s).suf_valid


def test_support_transfer_requires_source_identity(sample):
    q,s=sample
    unrelated=Evidence.make("Different subject",q.evidence[0].text+" More text.","different/source")
    pool=replace(s.pool,candidates=(unrelated,))
    labels=labels_for(q,replace(s,pool=pool))
    assert not labels.rel_valid[0] and labels.gains[0]==0
    chunk=Evidence("retrieved_chunk",q.evidence[0].title,q.evidence[0].text,q.evidence[0].source)
    labels=labels_for(q,replace(s,pool=replace(s.pool,candidates=(chunk,))))
    assert labels.rel_valid[0] and labels.gains[0]>.0
    with pytest.raises(ValueError): labels_for(q,replace(s,conditions="Use a different time period"))


def test_sufficiency_and_relevance_input_leakage(sample):
    q, s = sample
    choice, suf, rel = requests(s)
    text = json.dumps(suf)
    for e in q.evidence: assert e.text not in text
    assert q.answer not in text and "u0" not in text
    s = s.read(q.evidence[0].id)
    text = json.dumps(requests(s)[1])
    assert q.evidence[0].text in text and q.evidence[1].text not in text
    assert "selected_evidence" not in json.dumps(rel)


def test_family_split_and_gold_injection_barrier(sample):
    q, s = sample
    variant = replace(q, id="variant", split="dev")
    assigned = split_questions([q, variant])
    assert {x.split for x in assigned} == {"test"}
    with pytest.raises(ValueError): training_pool(assigned[0])
    with pytest.raises(ValueError): require_retrieved(s.pool)
    with pytest.raises(ValueError): audit_splits([q, replace(variant, split="validation")])


def test_pool_immutability_and_dedup(sample, tmp_path):
    q, s = sample
    cache = PoolCache(tmp_path); cache.write(s.pool)
    assert cache.read(q.id) == s.pool
    with pytest.raises(ValueError): cache.write(replace(s.pool, candidates=s.pool.candidates[:-1]))
    with pytest.raises(ValueError): replace(s.pool, candidates=s.pool.candidates+(s.pool.candidates[0],))
    assert s.read(q.evidence[0].id).pool.fingerprint == s.pool.fingerprint
    assert len(training_pool(q, k=64).candidates) == 6


def test_full_candidate_normalization_checkpoint_gradient_and_permutation(sample):
    torch.set_num_threads(2)
    q, s = sample
    model = tiny_model(); model.train()
    p = model(s); y = labels_for(q, s)
    torch.testing.assert_close(p.p.sum(), torch.tensor(1.))
    c, _ = task_loss([p], [y]); c["task"].backward()
    for name in ("core.adapter.embedding.weight", "core.head.interactor.blocks.0.attention.in_proj_weight"):
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    model.eval()
    s2 = replace(s, pool=replace(s.pool, candidates=tuple(reversed(s.pool.candidates))))
    with torch.no_grad():
        a, b = model(s, include_rel=False), model(s2, include_rel=False)
        torch.testing.assert_close(a.p, b.p.flip(0), atol=2e-6, rtol=1e-5)


def test_k256_uses_one_interactor_and_one_softmax():
    from openjev.models.decision.head import DecisionHead
    from openjev.models.decision.schema import Decision, Option
    decision = Decision("large", "state", "question", tuple(Option(str(i), str(i)) for i in range(256)))
    assert len(decision.options) == 256
    head = DecisionHead(16, set_dim=16, set_heads=4, set_layers=1, ffn_dim=32)
    features = torch.randn(1, 256, 16, requires_grad=True)
    out = head(torch.randn(1, 16), features, torch.ones(1, 256, dtype=torch.bool))
    torch.testing.assert_close(out.probabilities.sum(), torch.tensor(1.))
    (-out.log_probs[0, 0]).backward()
    assert features.grad[0, -1].abs().sum() > 0  # distant candidate retained in gradient graph


def make_pred(p, h, ids):
    return Prediction(ids, torch.tensor(p).log().requires_grad_(), torch.tensor([1-h,h]).log().requires_grad_(),
                      torch.full((len(ids), 2), math.log(.5), requires_grad=True))


def test_stability_id_alignment_tolerance_and_real_change_mask(sample):
    q, s = sample; y = labels_for(q,s); ids = y.candidate_ids
    a = make_pred([.5,.5,0.000001,0.000001,0.000001,0.000001], .2, ids)
    b = make_pred([.5001,.4999,0.000001,0.000001,0.000001,0.000001], .22, ids)
    loss, n = stability_loss([a],[b],[y],[s],[s]); assert float(loss) == 0 and n["stab_rank"] == 1
    c = make_pred([.8,.2,0.000001,0.000001,0.000001,0.000001], .4, ids)
    expected = max(math.sqrt(.18)-.05,0)**2 + (.2-.05)**2
    loss, _ = stability_loss([a],[c],[y],[s],[s]); assert float(loss) == pytest.approx(expected)
    loss.backward(); assert c.log_p.grad.abs().sum() > 0
    loss, n = stability_loss([a],[c],[y],[s],[replace(s,budget=2)])
    assert float(loss) == 0 and sum(n.values()) == 0
    d = make_pred(list(reversed(c.p.detach().tolist())), .4, tuple(reversed(ids)))
    loss, _ = stability_loss([a],[d],[y],[s],[s]); assert float(loss) == pytest.approx(expected)


def test_separate_mask_denominators(sample):
    q,s=sample; y=labels_for(q,s); p=make_pred([1/6]*6,.25,y.candidate_ids)
    unknown=replace(y, rank_valid=False,suf_valid=False,rel_valid=(False,)*6)
    c,n=task_loss([p,p],[y,unknown]); assert n=={"rank":1,"suf":1,"rel":6}
    assert float(c["suf"])==pytest.approx(-math.log(.75))


def test_full_trajectory_and_budget_no_stop(sample):
    q,s=sample
    class Fixed:
        def __call__(self,state,view,include_rel=False):
            return make_pred([1/len(state.candidates)]*len(state.candidates),.25,tuple(e.id for e in state.candidates))
    t=Trajectory(s,(s.candidates[0].id,"STOP"),"stop")
    value=trajectory_log_probability(Fixed(),t)
    assert float(value)==pytest.approx(math.log(.75)+math.log(1/6)+math.log(.25))
    exhausted=replace(s,budget=1)
    t=Trajectory(exhausted,(exhausted.candidates[0].id,),"budget")
    assert float(trajectory_log_probability(Fixed(),t))==pytest.approx(math.log(.75)+math.log(1/6))
    with pytest.raises(ValueError): Trajectory(exhausted,(exhausted.candidates[0].id,"STOP"),"stop").replay()


def test_dpo_reference_is_detached_and_direction_correct():
    chosen=torch.tensor(-1.,requires_grad=True); rejected=torch.tensor(-3.,requires_grad=True)
    ref=torch.tensor(-2.,requires_grad=True)
    loss=dpo_loss(chosen,rejected,ref,ref); loss.backward()
    assert chosen.grad<0 and rejected.grad>0 and ref.grad is None


def test_paired_statistics_reject_different_caches():
    a=[dict(question_id="q",group_id="g",pool_hash="a",score=1.)]
    b=[dict(question_id="q",group_id="g",pool_hash="b",score=0.)]
    with pytest.raises(ValueError): paired_bootstrap(a,b,"score")
    b[0]["pool_hash"]="a"; assert paired_bootstrap(a,b,"score")["difference"]==1.
    assert js_divergence([.5,.5],[.5,.5])==0
