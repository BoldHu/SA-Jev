from dataclasses import replace
import torch
from sa_jev.smoke import fixture_questions
from sa_jev.schema import Pool
from sa_jev.data import task_states
from sa_jev.model import tiny_model
from sa_jev.training import fit, TrainConfig


def test_optimizer_rng_resume_matches_uninterrupted(tmp_path):
    torch.set_num_threads(2)
    questions=fixture_questions()
    train=task_states(questions[0],Pool(questions[0].id,questions[0].evidence[:3],"fixture"),budget=2)
    val=task_states(questions[2],Pool(questions[2].id,questions[2].evidence[:3],"fixture"),budget=2)
    cfg=TrainConfig(stage=2,dpo_weight=0.,accumulation=1,max_steps=2,save_every=1,validate_every=2,stability_warmup_steps=1)
    uninterrupted=tiny_model()
    fit(uninterrupted,train,val,cfg,tmp_path/'full')
    resumed=tiny_model()
    fit(resumed,train,val,cfg,tmp_path/'resumed',resume=tmp_path/'full/step_000001')
    for (name,a),(_,b) in zip(uninterrupted.named_parameters(),resumed.named_parameters()):
        torch.testing.assert_close(a,b,atol=0,rtol=0,msg=name)
