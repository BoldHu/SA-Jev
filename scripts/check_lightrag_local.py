"""Native LightRAG ingestion/mix retrieval with deterministic fixture LLM/embeddings.

No external service is called and this is not a scientific retrieval baseline.
"""
import asyncio
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from sa_jev.retrieval import ingest_corpus, retrieve_once, PoolCache
from sa_jev.schema import Evidence
from sa_jev.io import write_json


async def embedding(texts):
    result = np.zeros((len(texts), 32),dtype=np.float32)
    for i,text in enumerate(texts):
        for word in re.findall(r"\w+",text.lower()):
            result[i,int(hashlib.sha256(word.encode()).hexdigest()[:8],16)%32]+=1
    return result / np.maximum(np.linalg.norm(result,axis=1,keepdims=True),1e-6)


async def completion(prompt, system_prompt=None, **kwargs):
    if kwargs.get('keyword_extraction') or 'high_level_keywords' in prompt:
        return json.dumps({"high_level_keywords":["birthplace"],"low_level_keywords":["Ada","London"]})
    if '---Input Text---' in prompt:
        return ('entity<|#|>Ada<|#|>Person<|#|>Ada was born in London.\n'
                'entity<|#|>London<|#|>Location<|#|>London is a city in England.\n'
                'relation<|#|>Ada<|#|>London<|#|>birthplace<|#|>Ada was born in London.\n<|COMPLETE|>')
    return 'Ada was born in London. London is in England.'


async def main():
    root=Path('outputs/lightrag_local_v2'); root.mkdir(parents=True,exist_ok=True)
    rag=LightRAG(working_dir=str(root/'index'), llm_model_func=completion,
        embedding_func=EmbeddingFunc(embedding_dim=32,max_token_size=2048,func=embedding),
        entity_extract_max_gleaning=0, cosine_better_than_threshold=-1.0, chunk_token_size=128, chunk_overlap_token_size=16)
    await rag.initialize_storages()
    try:
        docs=[Evidence.make('Ada','Ada was born in London. London is a city in England.','fixture://ada'),
              Evidence.make('Museum','A museum in London has paintings.','fixture://museum')]
        fingerprint=await ingest_corpus(rag,docs,root/'manifest.json',{'fixture':True})
        pool=await retrieve_once(rag,'fixture-query','Where was Ada born?',8,PoolCache(root/'pools'),
                                 corpus_fingerprint=fingerprint,mode='mix',source_registry={e.id:e for e in docs})
        assert pool.candidates and all(e.id and e.source for e in pool.candidates)
        assert len({e.id for e in pool.candidates})==len(pool.candidates)
        assert {e.source for e in pool.candidates} <= {e.source for e in docs}
        write_json(root/'verification.json',{'native_ingestion':True,'native_mix_retrieval':True,
                   'returned_candidates':len(pool.candidates),'fixed_cache':True,
                   'test_only':True,'llm_and_embeddings':'deterministic fixtures; not a research baseline'})
        print('Native LightRAG fixture integration passed:',len(pool.candidates),'candidates')
    finally:
        await rag.finalize_storages()


if __name__=='__main__': asyncio.run(main())
