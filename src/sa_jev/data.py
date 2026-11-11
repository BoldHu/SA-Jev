"""Dataset adapters, question-family splits, and bounded task-state generation."""
from __future__ import annotations

from dataclasses import replace
import random
from .io import digest
from .schema import Evidence, Question, Pool, State, TaskExample, deduplicate, labels_for, normalize


def _context_pairs(row):
    context = row["context"]
    return list(zip(context["title"], context["sentences"])) if isinstance(context, dict) else context


def _support_pairs(row):
    facts = row["supporting_facts"]
    return list(zip(facts["title"], facts["sent_id"])) if isinstance(facts, dict) else facts


def hotpot(row, split="train", dataset="hotpotqa"):
    qid = dataset + ":" + str(row.get("_id", row.get("id")))
    facts = [(str(title), int(index)) for title, index in _support_pairs(row)]
    units = tuple(digest([title, index])[:20] for title, index in facts)
    lookup = {pair: uid for pair, uid in zip(facts, units)}
    paragraphs, coverage, unit_text = [], {}, {}
    for title, sentences in _context_pairs(row):
        e = Evidence.make(title, " ".join(sentences), f"{dataset}/{title}")
        paragraphs.append(e)
        covered = []
        for i, sentence in enumerate(sentences):
            if (title, i) in lookup:
                uid = lookup[(title, i)]
                covered.append(uid)
                unit_text[uid] = sentence
        coverage[e.id] = tuple(covered)
    evidence = deduplicate(paragraphs)
    return Question(qid, qid, dataset, split, row["question"], row.get("answer", ""), evidence,
                    units, coverage, unit_text, tuple(e.id for e in evidence if not coverage[e.id]),
                    len(unit_text) == len(units) and bool(units))


def musique(row, split="train"):
    if not row.get("answerable", True):
        raise ValueError("MuSiQue unanswerable records are excluded; labels are context-bound")
    if "paragraphs" not in row:
        return musique_legacy(row, split)
    qid = "musique:" + str(row["id"])
    # Shared decomposition question IDs link generated multi-hop variants. The
    # connected-component split below also joins identical normalized questions.
    group = "musique:" + str(row.get("original_question_id", row["id"]))
    steps = row.get("question_decomposition", [])
    units = tuple("step:" + str(s["id"]) for s in steps)
    paragraphs, coverage, unit_text = [], {}, {}
    for p in row["paragraphs"]:
        e = Evidence.make(p["title"], p["paragraph_text"], f"musique/{p['title']}")
        paragraphs.append(e)
        cov = tuple("step:" + str(s["id"]) for s in steps if int(s["paragraph_support_idx"]) == int(p["idx"]))
        coverage[e.id] = cov
        for uid in cov:
            unit_text[uid] = p["paragraph_text"]
    evidence = deduplicate(paragraphs)
    return Question(qid, group, "musique", split, row["question"], row["answer"], evidence,
                    units, coverage, unit_text,
                    tuple(e.id for e in evidence if not coverage[e.id]),
                    bool(units) and len(unit_text) == len(units), tuple(row.get("answer_aliases", [])))


def musique_legacy(row, split="train"):
    """Official author's raw experiment format; supervision stays in decomposition."""
    qid = "musique:" + str(row["id"])
    steps = row["decomposed_instances"]
    units, support = [], {}
    for index, step in enumerate(steps):
        uid = "step:" + str(index)
        units.append(uid)
        for p in step["contexts"]:
            if p.get("is_supporting"):
                support.setdefault(normalize(p["paragraph_text"]), []).append(uid)
    evidence, coverage, unit_text, negatives = [], {}, {}, []
    for p in row["contexts"]:
        e = Evidence.make(p["wikipedia_title"], p["paragraph_text"], f"musique/{p['wikipedia_title']}")
        evidence.append(e)
        cov = tuple(support.get(normalize(p["paragraph_text"]), ()))
        if cov or p.get("is_supporting") is False:
            coverage[e.id] = cov
        if p.get("is_supporting") is False: negatives.append(e.id)
        for uid in cov: unit_text[uid] = p["paragraph_text"]
    return Question(qid, qid, "musique", split, row["composed_question_text"], row["answer_text"],
                    deduplicate(evidence), tuple(units), coverage, unit_text, tuple(negatives), len(unit_text)==len(units))


def twowiki(row, split="test"):
    # Sentence support is the proxy, not the relation triples or the answer string.
    return hotpot(row, split=split, dataset="2wiki")


def split_questions(questions, family_links=(), seed=17, validation_fraction=.1, heldout_fraction=.05):
    """Assign connected families before sampling states, views or trajectories.

    Official development records stay test; official training families containing
    a development duplicate are excluded from training as a complete component.
    family_links may contain externally supplied original/variant question IDs.
    """
    qs = list(questions)
    parent = {q.id: q.id for q in qs}
    if len(parent) != len(qs):
        raise ValueError("duplicate question IDs")

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def join(a, b):
        if a in parent and b in parent:
            a, b = find(a), find(b)
            parent[max(a, b)] = min(a, b)

    seen = {}
    for q in qs:
        for key in ("text:" + normalize(q.question), "group:" + q.group_id):
            if key in seen:
                join(q.id, seen[key])
            seen[key] = q.id
    for a, b in family_links:
        join(a, b)
    test_groups = {find(q.id) for q in qs if q.split in {"dev", "validation", "test"}}
    output = []
    for q in qs:
        group = find(q.id)
        x = int(digest([seed, group])[:12], 16) / 16**12
        split = "test" if group in test_groups else (
            "heldout" if x < heldout_fraction else "validation" if x < heldout_fraction + validation_fraction else "train")
        output.append(replace(q, group_id=group, split=split))
    return output


def audit_splits(questions, examples=(), pairs=(), heldout_groups=()):
    groups, texts = {}, {}
    for q in questions:
        for mapping, key in ((groups, q.group_id), (texts, normalize(q.question))):
            if key in mapping and mapping[key] != q.split:
                raise ValueError("question family or text crosses splits")
            mapping[key] = q.split
    reserved = set(heldout_groups)
    for x in examples:
        if groups.get(x.state.group_id) != x.split:
            raise ValueError("state split differs from question family")
        if x.split in {"train", "validation"} and x.state.group_id in reserved:
            raise ValueError("independent perturbation test leaked into training or tuning")
    for pair in pairs:
        if groups.get(pair.chosen.initial.group_id) != "train":
            raise ValueError("DPO pair must come from training family")
    return {s: sum(q.split == s for q in questions) for s in ("train", "validation", "test", "heldout")}


def sample_training(questions, counts=None, seed=17):
    counts = counts or {"hotpotqa": 8000, "musique": 4000}
    rng = random.Random(seed)
    result = []
    for dataset, count in counts.items():
        eligible = [q for q in questions if q.split == "train" and q.dataset == dataset]
        rng.shuffle(eligible)
        result.extend(eligible[:count])
    return result


def training_pool(q, retrieved=(), k=16):
    if q.split != "train":
        raise ValueError("gold-supported training pools are forbidden in evaluation")
    positives = [e for e in q.evidence if q.coverage.get(e.id)]
    negatives = [e for e in q.evidence if not q.coverage.get(e.id)]
    pool = deduplicate([*positives, *retrieved, *negatives])[:k]
    # IDs and order reveal no gold status. Selection labels stay out of inputs.
    pool = tuple(sorted(pool, key=lambda e: digest([q.id, e.id, "pool-order"])))
    return Pool(q.id, pool, "training_supervision", {"requested_k": k, "actual_k": len(pool),
                "gold_injected": True, "purpose": "training_only"})


def task_states(q, pool, budget=4, seed=17, max_states=3):
    if not 1 <= max_states <= 3:
        raise ValueError("protocol permits at most three task states per question")
    rng = random.Random(digest([seed, q.id]))
    state = State(q.id, q.group_id, q.question, pool, budget=budget)
    trajectory = [state]
    # Adjacent true evidence acquisitions retained for task supervision.
    while state.budget and state.candidates:
        labels = labels_for(q, state)
        if labels.suf_valid and labels.sufficient:
            break
        gains = dict(zip(labels.candidate_ids, labels.gains))
        best = max(gains.values(), default=0)
        candidates = [e for e in state.candidates if gains[e.id] == best]
        state = state.read(rng.choice(candidates).id)
        trajectory.append(state)
    indices = list(range(len(trajectory))) if len(trajectory) <= max_states else sorted(
        {0, rng.randrange(1, len(trajectory) - 1), len(trajectory) - 1})[:max_states]
    return [TaskExample(trajectory[i], labels_for(q, trajectory[i]), q.split) for i in indices]


def build_training_states(questions, retrieved=None, seed=17, budget=4, small_k=False):
    from tqdm.auto import tqdm
    rng, rows = random.Random(seed), []
    for q in tqdm(questions, desc="Build task states", unit="question", mininterval=5):
        k = 8 if small_k else rng.choices([8, 16, 32, 64], [.4, .3, .2, .1])[0]
        candidates = (retrieved or {}).get(q.id, ())
        pool = training_pool(q, candidates, k)
        rows.extend(task_states(q, pool, budget, seed))
    return rows


def overlap_audit(questions, public_records):
    """Exact/contained normalized question matches; not a proof of no pretraining exposure."""
    needles = {normalize(q.question): q.id for q in questions}
    prefixes = {}
    short_questions = []
    for question in needles:
        tokens = question.split()
        if len(tokens) >= 4:
            prefixes.setdefault(tuple(tokens[:4]), []).append(question)
        else:
            short_questions.append(question)
    hits, count = [], 0
    for record in public_records:
        count += 1
        # OpenJevData may embed source questions inside state or instructions.
        fields = []
        def collect(x):
            if isinstance(x, str): fields.append(normalize(x))
            elif isinstance(x, dict):
                for v in x.values(): collect(v)
            elif isinstance(x, list):
                for v in x: collect(v)
        collect(record)
        haystack = "\n".join(fields)
        tokens = haystack.split(); possible = set()
        for i in range(len(tokens)-3): possible.update(prefixes.get(tuple(tokens[i:i+4]), ()))
        possible.update(short_questions)
        for question in possible:
            if len(question) > 15 and question in haystack:
                hits.append({"question_id": needles[question], "record_index": count - 1, "match": "normalized_containment"})
        if count % 1000 == 0: print(f"OpenJevData overlap audit: {count} records scanned, {len(hits)} matches", flush=True)
    return {"records_scanned": count, "matches": hits,
            "limitation": "No match does not establish unseen generalization; paraphrases and undocumented sources remain unknown."}
