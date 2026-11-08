"""One real query on an already imported public graph; no extraction or answer generation."""
import argparse
import asyncio
from pathlib import Path
import time

from sa_jev.environment import load_environment
from sa_jev.experiments import load_config
from sa_jev.public_indexes import open_public_index
from sa_jev.io import read_json, write_json
from sa_jev.retrieval import retrieve_once, PoolCache, _redact_error
from sa_jev.schema import Evidence


async def main(root):
    import torch
    torch.set_num_threads(2)
    root = Path(root)
    load_environment(Path('.env'))
    config = load_config(); config['retrieval']['embedding_device'] = 'cpu'
    rag = await open_public_index(root, 'hotpotqa', config)
    registry = {k: Evidence(**v) for k,v in read_json(root/'index/hotpotqa/source_registry.json').items()}
    published = read_json(root/'index/hotpotqa/lightrag_contexts.json')
    queries = read_json(root.parent/'hotpotqa_queries.json')
    q = next(q for q in queries if q['_id'] in published)
    started = time.perf_counter()
    try:
        pool = await retrieve_once(rag, 'live:'+q['_id'], q['question'], 256,
                    PoolCache(root/'live_pools'), corpus_fingerprint='public_live_check_v1', chunk_registry=registry,
                    extra={'max_total_tokens':180000, 'max_entity_tokens':20000, 'max_relation_tokens':20000})
        result = {'success': bool(pool.candidates), 'question_id': q['_id'], 'requested_k':256,
                  'returned_k':len(pool.candidates), 'seconds':time.perf_counter()-started,
                  'query_model':config['retrieval']['llm_model'], 'source_traceability':True,
                  'document_extraction_api_calls':0, 'scope':'one live keyword query; no QA or paper performance'}
    except Exception as error:
        result = {'success':False, 'error':_redact_error(error)}
    finally:
        await rag.finalize_storages()
    write_json(root/'live_verification.json', result)
    print(result)
    if not result['success']: raise SystemExit(1)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True)
    asyncio.run(main(p.parse_args().root))
