"""Exercise the actual Qwen3.5 adapter + PEFT + OpenJev head with tiny random weights.

This checks API and gradient compatibility; it does not validate OpenJev-4B/FLa
weights or the A100 memory budget. No Hugging Face model download is performed.
"""
import argparse
from pathlib import Path
from time import perf_counter
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
from peft import LoraConfig, get_peft_model
from openjev.models.qwen35.adapter import Qwen35Adapter
from openjev.models.decision.head import DecisionHead
from openjev.models.decision.model import OpenJevDecisionModel
from openjev.models.decision.serializer import DecisionSerializer
from sa_jev.model import SAJev, TinyTokenizer
from sa_jev.smoke import fixture_questions
from sa_jev.schema import Pool, State, labels_for
from sa_jev.losses import task_loss
from sa_jev.io import write_json


def check(device="cpu", output="outputs/qwen_lora_check.json", compare_batching=False):
    torch.set_num_threads(2); torch.manual_seed(17)
    config = Qwen3_5TextConfig(vocab_size=2048, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2, linear_num_value_heads=4,
        layer_types=["linear_attention", "full_attention"], pad_token_id=0)
    lora = dict(r=4, lora_alpha=8, target_modules="all-linear", lora_dropout=0., bias="none")
    backbone = get_peft_model(Qwen3_5TextModel(config), LoraConfig(**lora))
    core = OpenJevDecisionModel(Qwen35Adapter(backbone), DecisionHead(32, set_dim=32, set_heads=4,
                               set_layers=1, ffn_dim=64), branch_microbatch_size=1, checkpoint_branches=True).to(device)
    tokenizer = TinyTokenizer(); model = SAJev(core, tokenizer, DecisionSerializer(tokenizer),
                                              base_identity="TEST_ONLY_tiny_Qwen3.5", lora_config=lora)
    q = fixture_questions()[0]; state = State(q.id,q.group_id,q.question,Pool(q.id,q.evidence[:3],"fixture"),budget=3)
    comparison = None
    if compare_batching:
        measurements = []
        baseline = None
        other_q = fixture_questions()[1]
        other = State(other_q.id, other_q.group_id, other_q.question,
                      Pool(other_q.id, other_q.evidence[:4], "fixture"), budget=3)
        other = other.read(other.candidates[0].id)
        states = [state, other]
        labels = [labels_for(q, state), labels_for(other_q, other)]
        model.train()
        for branches, decisions, state_batch in ((1, 1, 1), (2, 8, 2), (4, 8, 2)):
            core.branch_microbatch_size = branches
            model.decision_batch_size = decisions
            model.zero_grad(set_to_none=True)
            if str(device).startswith('cuda'): torch.cuda.synchronize()
            started = perf_counter()
            predictions = ([model(s) for s in states] if state_batch == 1 else model.forward_batch(states))
            values, _ = task_loss(predictions, labels)
            values['task'].backward()
            if str(device).startswith('cuda'): torch.cuda.synchronize()
            seconds = perf_counter()-started
            tensors = {f'{index}/{key}': getattr(prediction, key).detach().clone()
                       for index, prediction in enumerate(predictions) for key in ('log_p', 'log_h', 'log_rel')}
            tensors['loss'] = values['task'].detach().clone()
            gradient = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
            max_error = 0.
            if baseline is None:
                baseline = tensors, gradient
            else:
                for key in tensors:
                    torch.testing.assert_close(tensors[key], baseline[0][key], atol=2e-5, rtol=2e-4)
                assert gradient.keys() == baseline[1].keys()
                for name in gradient:
                    torch.testing.assert_close(gradient[name], baseline[1][name], atol=2e-5, rtol=2e-3, msg=name)
                    max_error = max(max_error, float((gradient[name]-baseline[1][name]).abs().max()))
            measurements.append(dict(branch_microbatch=branches, decision_batch_size=decisions, state_batch_size=state_batch,
                                     forward_backward_seconds=seconds, max_gradient_absolute_error=max_error))
            print('[batch check]', measurements[-1], flush=True)
        comparison = {'predictions_loss_gradients_match': True, 'measurements': measurements,
                      'timing_scope': 'tiny random model, includes warmup; not a 4B speed estimate'}
        model.zero_grad(set_to_none=True)
    model.train(); pred=model(state); components,_=task_loss([pred],[labels_for(q,state)])
    components["task"].backward()
    grads={name:float(p.grad.abs().sum()) for name,p in model.named_parameters() if p.requires_grad and p.grad is not None}
    assert any("lora_B" in name and value>0 for name,value in grads.items())
    assert any("interactor" in name and value>0 for name,value in grads.items())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    before={name:p.detach().clone() for name,p in model.named_parameters() if "lora_B" in name}
    torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001).step()
    assert any(not torch.equal(before[name],p) for name,p in model.named_parameters() if name in before)
    result={"device":device,"model":"random tiny Qwen3.5, mixed linear/full attention", "lora_backward":True,
            "interactor_backward":True,"optimizer_update":True,"task_loss":float(components['task'].detach()),
            "parameters":sum(p.numel() for p in model.parameters()), "trained_OpenJev4B":False,
            "delta_kernel": next((f'{m.chunk_gated_delta_rule.__module__}.{m.chunk_gated_delta_rule.__name__}'
                                  for m in model.modules() if hasattr(m, 'chunk_gated_delta_rule')), None),
            "batching_comparison": comparison}
    write_json(output,result); print(result)


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("--device",default="cpu");p.add_argument("--output",default="outputs/qwen_lora_check.json")
    p.add_argument('--compare-batching', action='store_true')
    args=p.parse_args();check(args.device,args.output,args.compare_batching)
