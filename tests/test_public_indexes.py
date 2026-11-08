import asyncio
import base64
from dataclasses import replace
import json
from pathlib import Path
import warnings
from zipfile import ZipFile

import networkx as nx
import numpy as np
import pytest

from sa_jev import public_indexes as public
from sa_jev.io import read_json, write_json
from sa_jev.retrieval import PoolCache, retrieve_once
from sa_jev.schema import Evidence


def test_duplicate_zip_paths_and_explicit_2wiki_repairs(tmp_path, monkeypatch):
    path = tmp_path/"release.zip"
    with warnings.catch_warnings(), ZipFile(path, "w") as z:
        warnings.simplefilter("ignore", UserWarning)
        for ds in public.DATASETS:
            cid, did = f"chunk-{ds}", f"doc-{ds}"
            graph = nx.Graph(); graph.add_node(ds, source_id=cid)
            entries = {name: {} for name in public.FILES}
            entries["graph_chunk_entity_relation.graphml"] = "\n".join(nx.generate_graphml(graph))
            entries["kv_store_full_docs.json"] = {did: {"content": ds+"\n\nbody"}}
            entries["kv_store_text_chunks.json"] = {cid: {"full_doc_id": did, "content": ds+" body"}}
            entries["vdb_chunks.json"] = {"embedding_dim": 384, "data": [{"__id__": cid}]}
            entries["vdb_entities.json"] = {"embedding_dim": 384, "data": [{"entity_name": ds}]}
            entries["vdb_relationships.json"] = {"embedding_dim": 384, "data": []}
            entries["lightrag_contexts.json"] = {ds: "published context"}
            for name, value in entries.items():
                content = value if isinstance(value, str) else json.dumps(value)
                if ds == "2wiki" and name in public.REPAIRS:
                    z.writestr("large_scale_lightrag/"+public.REPAIRS[name], content)
                    content = "{}"  # known incomplete canonical file must NOT be used
                z.writestr("large_scale_lightrag/"+name, content)
    monkeypatch.setattr(public, "SHA256", public.file_hash(path))
    manifest = public.import_public_indexes(path, tmp_path/"work")
    for ds in public.DATASETS:
        assert manifest["stats"][ds]["documents"] == 1
        registry = read_json(tmp_path/"work/index"/ds/"source_registry.json")
        assert registry[f"chunk-{ds}"]["title"] == ds
    assert public.import_public_indexes(path, tmp_path/"work") == manifest
    # A present file with missing vectors must never pass the content audit.
    bad = tmp_path/"work/index/hotpotqa/vdb_chunks.json"
    write_json(bad, {"embedding_dim": 384, "data": []})
    with pytest.raises(ValueError, match="incomplete"):
        public.audit_index(bad.parent)


def test_encoder_probe_rejects_same_dimension_wrong_model(tmp_path):
    a = np.zeros((3,384), dtype=np.float32); a[:,0] = 1
    write_json(tmp_path/"vdb_chunks.json", {"data": [{"__id__": str(i), "content": str(i)} for i in range(3)],
               "matrix": base64.b64encode(a.tobytes()).decode()})
    async def correct(texts): return a
    asyncio.run(public.verify_encoder(tmp_path, correct))
    async def wrong(texts): return np.roll(a, 1, axis=1)
    with pytest.raises(ValueError, match="compatibility"):
        asyncio.run(public.verify_encoder(tmp_path, wrong))


def test_public_retrieval_resolves_chunk_not_unknown_source(tmp_path):
    original = Evidence("chunk-x", "Title", "Text", "hotpotqa/Title")
    class Rag:
        calls = 0
        async def aquery_data(self, query, param):
            self.calls += 1
            return {"status": "success", "data": {"chunks": [
                {"chunk_id": "chunk-x", "content": "Text", "file_path": "unknown_source"}]}}
    rag = Rag(); cache = PoolCache(tmp_path)
    async def run():
        p = await retrieve_once(rag, "q", "question", 8, cache, corpus_fingerprint="release",
                                chunk_registry={"chunk-x": original})
        assert p.candidates == (original,)
        assert await retrieve_once(rag, "q", "question", 8, cache, corpus_fingerprint="release",
                                   chunk_registry={"chunk-x": original}) == p
        assert rag.calls == 1
        with pytest.raises(ValueError, match="pinned source"):
            await retrieve_once(rag, "bad", "question", 8, cache, corpus_fingerprint="release", chunk_registry={})
    asyncio.run(run())


def test_public_router_never_ingests_training_corpus(tmp_path, monkeypatch):
    from sa_jev import experiments
    calls = []
    async def fake(root, config, limit): calls.append((root,limit)); return "public"
    monkeypatch.setattr(public, "cache_public_pools", fake)
    assert asyncio.run(experiments.cache_pools(tmp_path, {"protocol": "public_lightrag_n500_v1"}, 3)) == "public"
    assert calls == [(tmp_path,3)]


def test_public_notebooks_are_isolated_and_never_rebuild():
    import ast
    for path in Path("notebooks").glob("*.ipynb"):
        nb = read_json(path)
        cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        assert "public_lightrag_v1" in cells[0]
        assert "configs/public_lightrag.yaml" in cells[0]
        for source in cells: ast.parse(source)
        if path.name.startswith("00"):
            joined = "\n".join(cells)
            assert "prepare_public_data" in joined
            assert "normalize_sources(" not in joined
            assert "download_2wiki(" not in joined


def test_full_question_protocol_keeps_500_tests_and_disjoint_roles(tmp_path, monkeypatch):
    from sa_jev import data
    from sa_jev.schema import Question
    from sa_jev.io import write_jsonl, read_jsonl
    assets, raw, root = tmp_path/"assets", tmp_path/"raw", tmp_path/"run"
    def adapter(ds):
        def convert(row, split="train"):
            e = Evidence.make(ds, "support", ds+"/source")
            return Question(ds+":"+row["id"], ds+":"+row["id"], ds, split,
                            row.get("question", ds+" question "+row["id"]), "answer", (e,),
                            ("u",), {e.id:("u",)}, {"u":"support"}, ())
        return convert
    for ds, name in zip(public.DATASETS, ("hotpot", "musique", "twowiki")):
        monkeypatch.setattr(data, name, adapter(ds))
        write_json(assets/f"{ds}_queries.json", [{"id":str(i)} for i in range(1000)])
        write_json(root/"index"/ds/"lightrag_contexts.json", {str(i):"cached context" for i in range(500)})
    for ds, name in (("hotpotqa","hotpot_train.jsonl"),("musique","musique_ans_train.jsonl")):
        rows = [{"id":"train"+str(i)} for i in range(10)]
        # A differently named training ID with identical test question must be excluded.
        rows.append({"id":"duplicate", "question":ds+" question 0"})
        write_jsonl(raw/name, rows)
    config={"seed":17,"data":{"train_counts":{"hotpotqa":2,"musique":2},
                              "validation_per_dataset":100,"heldout_per_dataset":300}}
    manifest=public.prepare_public_data(root,assets,raw,config,mode="full")
    assert manifest["splits"] == {"train":4,"validation":200,"test":1500,"heldout":600}
    rows=list(read_jsonl(root/"data/questions.jsonl"))
    assert not any(q["id"].endswith(":duplicate") for q in rows)
    assert all(q["split"]=="test" for q in rows if q["dataset"]=="2wiki")
    assert public.prepare_public_data(root,assets,raw,config,mode="full")==manifest


def test_completed_public_pools_reuse_without_loading_models(tmp_path, monkeypatch):
    from sa_jev.smoke import fixture_questions
    from sa_jev.schema import Pool
    from sa_jev.io import digest, write_jsonl
    from sa_jev import embeddings
    q=fixture_questions()[2]
    write_jsonl(tmp_path/"data/questions.jsonl",[q])
    write_json(tmp_path/"index/public_manifest.json", {"release":"fixture"})
    write_json(tmp_path/"data/data_manifest.json", {"question_fingerprint":"questions"})
    config={"retrieval":{},"data":{"main_k":8}}
    pool=Pool(q.id,q.evidence,"fixture",{})
    for name in ("pools","pools_max"): PoolCache(tmp_path/"data"/name).write(pool)
    marker={"complete":True,"questions":1,"question_fingerprint":"questions",
            "corpus_binding":digest({"release":"fixture"}),
            "config_fingerprint":digest({"retrieval":{},"main_k":8})}
    write_json(tmp_path/"data/pool_cache_manifest.json",marker)
    async def forbidden(**kwargs):raise AssertionError("must not load encoder or call service")
    monkeypatch.setattr(embeddings,"make_embedding_func",forbidden)
    assert asyncio.run(public.cache_public_pools(tmp_path,config))==marker
