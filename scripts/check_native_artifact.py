"""Validate the native artifact loader and LoRA/head checkpoint with tiny local weights."""
from pathlib import Path
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
from openjev.config import OpenJevConfig
from openjev.artifacts import save_decision_artifact
from openjev.models.qwen35.adapter import Qwen35Adapter
from openjev.models.decision.head import DecisionHead
from openjev.models.decision.model import OpenJevDecisionModel
from sa_jev.model import SAJev
from sa_jev.smoke import fixture_questions
from sa_jev.schema import State, Pool, labels_for
from sa_jev.losses import task_loss
from sa_jev.io import write_json


def main():
    torch.set_num_threads(2); torch.manual_seed(17)
    root=Path('outputs/native_artifact_check'); root.mkdir(parents=True,exist_ok=True)
    config=Qwen3_5TextConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,linear_key_head_dim=8,linear_value_head_dim=8,
        linear_num_key_heads=2,linear_num_value_heads=4,layer_types=['linear_attention','full_attention'],pad_token_id=0)
    head=DecisionHead(32,set_dim=32,set_heads=4,set_layers=1,ffn_dim=64)
    core=OpenJevDecisionModel(Qwen35Adapter(Qwen3_5TextModel(config).to(dtype=torch.bfloat16)),head)
    tok=Tokenizer(WordLevel({'[PAD]':0,'[UNK]':1,'True':2,'False':3,'Person':4,'City':5,'Country':6},unk_token='[UNK]'))
    tok.pre_tokenizer=Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=tok,pad_token='[PAD]',unk_token='[UNK]')
    oj=OpenJevConfig(hidden_size=32,head={k:v for k,v in head.config.items() if k!='input_dim'},
                    compute_contract='bf16-backbone-fp32-head-v1')
    artifact=root/'base'
    if not artifact.exists(): save_decision_artifact(core,tokenizer,oj,artifact,stage='test-only-random')
    model=SAJev.from_pretrained(str(artifact),device='cpu',lora_rank=4,lora_alpha=8)
    # The native pretrained head must be loaded, not silently reinitialized.
    from safetensors.torch import load_file
    reference=load_file(str(artifact/'decision_head.safetensors'))
    for k,v in model.core.head.state_dict().items(): torch.testing.assert_close(v,reference[k],atol=0,rtol=0)
    q=fixture_questions()[0]; state=State(q.id,q.group_id,q.question,Pool(q.id,q.evidence[:3],'fixture'),budget=3)
    loss,_=task_loss([model(state)],[labels_for(q,state)]); loss['task'].backward()
    assert any('lora_B' in n and p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters())
    torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001).step()
    model.save_trainable(root/'checkpoint')
    restored=SAJev.from_pretrained(str(artifact),device='cpu',lora_rank=4,lora_alpha=8)
    restored.load_trainable(root/'checkpoint');model.eval();restored.eval()
    with torch.no_grad():
        a,b=model(state,include_rel=False),restored(state,include_rel=False)
        torch.testing.assert_close(a.log_p,b.log_p,atol=0,rtol=0)
        torch.testing.assert_close(a.log_h,b.log_h,atol=0,rtol=0)
    result={'native_artifact_load':True,'original_head_preserved':True,'bf16_backbone_fp32_head':True,
            'lora_gradient':True,'adapter_checkpoint_roundtrip':True,'test_only':True,
            'trained_OpenJev4B':False,'fla_kernel':False}
    write_json(root/'verification.json',result);print(result)


if __name__=='__main__':main()
