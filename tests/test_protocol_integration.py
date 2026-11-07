from dataclasses import replace, asdict
import asyncio
import ast
import json
from pathlib import Path
import pytest
import torch
from sa_jev.smoke import fixture_questions
from sa_jev.schema import Pool, State, labels_for
from sa_jev.data import hotpot, musique, twowiki, task_states, overlap_audit
from sa_jev.retrieval import PoolCache, retrieve_once
from sa_jev.robustness import review_packets, validate_packet, evaluate_robustness
from sa_jev.model import tiny_model
from sa_jev.evaluation import rollout, OfficialJev


def test_dataset_formats_and_incomplete_support():
    raw={"_id":"a", "question":"Who founded the company?", "answer":"Ada",
         "context":[["Company",["Ada founded the company.","The company makes maps."]],["Park",["The park has trees."]]],
         "supporting_facts":[["Company",0]]}
    assert hotpot(raw).gold_complete and twowiki(raw).dataset=="2wiki"
    assert not hotpot({**raw,"supporting_facts":[["Missing",0]]}).gold_complete
    m={"id":"2hop_x", "question":"Where is Ada's company?", "answer":"City", "answerable":True,
       "paragraphs":[{"idx":0,"title":"Ada","paragraph_text":"Ada founded Company."},
                     {"idx":1,"title":"Company","paragraph_text":"Company is in City."}],
       "question_decomposition":[{"id":1,"paragraph_support_idx":0},{"id":2,"paragraph_support_idx":1}]}
    q=musique(m); assert len(q.units)==2 and q.gold_complete
    with pytest.raises(ValueError): musique({**m,"answerable":False})
    audit=overlap_audit([q],[{"state":"The question is: "+q.question}]); assert len(audit["matches"])==1


def test_light_rag_cache_query_once_and_traceability(tmp_path):
    pytest.importorskip("lightrag")
    class Rag:
        count=0
        async def aquery_data(self,query,param):
            self.count+=1
            assert param.mode=="mix" and not param.enable_rerank
            return {"status":"success", "data":{"chunks":[{"chunk_id":"chunk-1","content":"A fact.","file_path":"source"}]}}
    rag=Rag(); cache=PoolCache(tmp_path)
    async def run():
        a=await retrieve_once(rag,"q","Question",8,cache,corpus_fingerprint="corpus")
        b=await retrieve_once(rag,"q","Question",8,cache,corpus_fingerprint="corpus")
        return a,b
    a,b=asyncio.run(run()); assert a==b and rag.count==1
    assert a.candidates[0].id=="chunk-1" and a.candidates[0].source=="source"


def test_review_required_and_effective_labels(tmp_path):
    q=replace(fixture_questions()[0],split="heldout")
    state=State(q.id,q.group_id,q.question,Pool(q.id,q.evidence,"fixture"),budget=3)
    packets=review_packets([q],[state],tmp_path/"draft.jsonl")
    with pytest.raises(ValueError): validate_packet(packets[0],[q.group_id])
    for row in [*packets[0]["perturbations"],packets[0]["effective"]]:
        row.update(verified=True,reviewer="TEST_FIXTURE_REVIEW",rationale="Synthetic fixture has explicit independent facts.")
    clean,y,views,changed,yc=validate_packet(packets[0],[q.group_id])
    assert changed.selected_ids!=clean.selected_ids and yc.gains!=y.gains
    torch.set_num_threads(2)
    rows=evaluate_robustness(tiny_model(),packets,[q.group_id])
    assert len(rows)==4 and rows[-1]["kind"]=="effective" and "js" not in rows[-1]


def test_rollout_sufficiency_first_and_refusal():
    q=fixture_questions()[0]; s=State(q.id,q.group_id,q.question,Pool(q.id,q.evidence,"fixture"),budget=1)
    class Model:
        def __init__(self,h): self.h=h;self.types=[]
        def eval(self): return self
        def decision(self,body):
            spec=next(iter(body["questions"].values()));self.types.append(spec["type"])
            values=[1-self.h,self.h] if spec["type"]=="noul" else [1/len(spec["criteria"])]*len(spec["criteria"])
            return torch.tensor(values).log(),1,1
    model=Model(.9); r=rollout(model,s)
    assert model.types==["noul"] and r["termination"]=="stop"
    model=Model(.1); r=rollout(model,s)
    assert model.types==["noul","choice","noul"] and not r["accepted"] and r["qa"]["answer"]=="noanswer"
    assert r["trace"][-1]["action"]=="BUDGET"


def test_notebook_cells_parse_and_outputs_empty():
    import nbformat
    root=Path(__file__).resolve().parents[1]
    paths=list((root/"notebooks").glob("*.ipynb")); assert len(paths)==9
    for path in paths:
        nb=nbformat.read(path,as_version=4); nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type=="code":
                compile(cell.source,str(path),"exec",flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
                assert cell.outputs==[]
        assert "drive.mount" in nb.cells[1].source and "pull" in nb.cells[1].source
