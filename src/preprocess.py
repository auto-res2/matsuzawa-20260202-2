import random
from typing import Any, Dict, List, Tuple

from datasets import load_dataset

IDENTITY_PHRASES = [
    "As a Muslim person, ",
    "As a Black woman, ",
    "As a gay man, ",
    "As a disabled person, ",
]


def normalize_text(text: str, lowercase: bool, max_length: int) -> str:
    text = " ".join(str(text).strip().split())
    if lowercase:
        text = text.lower()
    if max_length > 0:
        tokens = text.split()
        if len(tokens) > max_length:
            text = " ".join(tokens[:max_length])
    return text


def make_probe_text(text: str) -> str:
    text = " ".join(str(text).strip().split())
    tokens = text.split()
    if len(tokens) > 12:
        text = " ".join(tokens[:12])
    frag = " ".join(text.split()[:5])
    return f"{text}. Not {frag}."


def make_identity_probe(text: str, seed: int) -> str:
    prefix = IDENTITY_PHRASES[seed % len(IDENTITY_PHRASES)]
    text = " ".join(str(text).strip().split())
    return prefix + text


def get_dataset_fields(name: str, cfg: Any) -> Tuple[str, str]:
    if name == cfg.dataset.name:
        text_field = cfg.dataset.text_field
        label_field = cfg.dataset.label_field
    else:
        default_fields = {
            "sst2": ("sentence", "label"),
            "yelp": ("text", "label"),
            "imdb": ("text", "label"),
        }
        if name not in default_fields:
            raise ValueError(f"Unknown dataset name {name}")
        text_field, label_field = default_fields[name]
    if text_field is None or label_field is None:
        raise ValueError(f"Missing text or label field for dataset {name}")
    return text_field, label_field


def to_examples(
    dataset,
    text_field: str,
    label_field: str,
    source: str,
    lowercase: bool,
    max_length: int,
) -> List[Dict[str, Any]]:
    examples = []
    for ex in dataset:
        if text_field not in ex:
            raise KeyError(f"Missing text field {text_field} for source {source}")
        if label_field not in ex:
            raise KeyError(f"Missing label field {label_field} for source {source}")
        text = normalize_text(ex[text_field], lowercase, max_length)
        label = int(ex[label_field])
        examples.append({"text": text, "label": label, "source": source})
    return examples


def sample(ds, split: str, size: int, seed: int):
    if size <= 0:
        return []
    subset = ds[split].shuffle(seed=seed)
    return subset.select(range(min(size, len(subset))))


def allocate_audit_counts(audit_total: int) -> Dict[str, int]:
    targets = {"yelp": 80, "imdb": 80, "sst_probe": 80, "identity": 40}
    total_target = sum(targets.values())
    if audit_total <= 0:
        return {k: 0 for k in targets}
    if audit_total >= total_target:
        return targets
    ratios = {k: v / total_target for k, v in targets.items()}
    counts = {k: int(round(audit_total * ratio)) for k, ratio in ratios.items()}
    if audit_total >= len(targets):
        counts = {k: max(1, v) for k, v in counts.items()}
    current = sum(counts.values())
    keys_by_target = sorted(targets.keys(), key=lambda k: targets[k], reverse=True)
    idx = 0
    while current < audit_total:
        counts[keys_by_target[idx % len(keys_by_target)]] += 1
        current += 1
        idx += 1
    idx = 0
    while current > audit_total:
        key = keys_by_target[idx % len(keys_by_target)]
        if counts[key] > 0:
            counts[key] -= 1
            current -= 1
        idx += 1
    return counts


def prepare_datasets(cfg: Any, seed: int) -> Dict[str, Any]:
    cache_dir = cfg.cache_dir
    sst2 = load_dataset("glue", "sst2", cache_dir=cache_dir)
    yelp = load_dataset("yelp_polarity", cache_dir=cache_dir)
    imdb = load_dataset("stanfordnlp/imdb", cache_dir=cache_dir)

    splits = cfg.dataset.splits
    opt_size = int(splits.opt_size)
    audit_total = int(splits.audit_total)
    audit_disc = int(splits.audit_disc)
    audit_conf = int(splits.audit_conf)
    eval_size = int(splits.eval_size)

    if audit_disc + audit_conf != audit_total:
        audit_disc = audit_total // 2
        audit_conf = audit_total - audit_disc

    lowercase = bool(cfg.dataset.preprocessing.lowercase)
    max_length = int(cfg.dataset.preprocessing.max_length)

    sst_text_field, sst_label_field = get_dataset_fields("sst2", cfg)
    yelp_text_field, yelp_label_field = get_dataset_fields("yelp", cfg)
    imdb_text_field, imdb_label_field = get_dataset_fields("imdb", cfg)

    d_opt_raw = sample(sst2, "train", opt_size, seed + 0)
    d_opt = to_examples(d_opt_raw, sst_text_field, sst_label_field, "sst2", lowercase, max_length)

    eval_sst = to_examples(
        sample(sst2, "validation", eval_size, seed + 1),
        sst_text_field,
        sst_label_field,
        "sst2",
        lowercase,
        max_length,
    )
    eval_yelp = to_examples(
        sample(yelp, "test", eval_size, seed + 2),
        yelp_text_field,
        yelp_label_field,
        "yelp",
        lowercase,
        max_length,
    )
    eval_imdb = to_examples(
        sample(imdb, "test", eval_size, seed + 3),
        imdb_text_field,
        imdb_label_field,
        "imdb",
        lowercase,
        max_length,
    )

    audit_counts = allocate_audit_counts(audit_total)

    yelp_audit = to_examples(
        sample(yelp, "train", audit_counts["yelp"], seed + 4),
        yelp_text_field,
        yelp_label_field,
        "yelp",
        lowercase,
        max_length,
    )
    imdb_audit = to_examples(
        sample(imdb, "train", audit_counts["imdb"], seed + 5),
        imdb_text_field,
        imdb_label_field,
        "imdb",
        lowercase,
        max_length,
    )
    sst_probe_src = sample(sst2, "train", audit_counts["sst_probe"], seed + 6)

    probe_examples = []
    for ex in sst_probe_src:
        probe_text = make_probe_text(ex[sst_text_field])
        probe_examples.append(
            {
                "text": normalize_text(probe_text, lowercase, max_length),
                "label": int(ex[sst_label_field]),
                "source": "sst2_probe",
            }
        )

    identity_examples = []
    for idx, ex in enumerate(list(sst_probe_src)[: audit_counts["identity"]]):
        identity_text = make_identity_probe(ex[sst_text_field], idx)
        identity_examples.append(
            {
                "text": normalize_text(identity_text, lowercase, max_length),
                "label": int(ex[sst_label_field]),
                "source": "identity_probe",
            }
        )

    audit_pool = yelp_audit + imdb_audit + probe_examples + identity_examples
    rng = random.Random(seed)
    rng.shuffle(audit_pool)
    audit_pool = audit_pool[:audit_total]

    audit_disc = min(audit_disc, len(audit_pool))
    audit_conf = min(audit_conf, len(audit_pool) - audit_disc)
    audit_disc_set = audit_pool[:audit_disc]
    audit_conf_set = audit_pool[audit_disc : audit_disc + audit_conf]

    eval_probe_src = sample(sst2, "validation", min(40, eval_size), seed + 7)
    eval_probes: List[Dict[str, Any]] = []
    for idx, ex in enumerate(eval_probe_src):
        neg_text = make_probe_text(ex[sst_text_field])
        eval_probes.append(
            {
                "text": normalize_text(neg_text, lowercase, max_length),
                "label": int(ex[sst_label_field]),
                "source": "eval_probe",
            }
        )
        identity_text = make_identity_probe(ex[sst_text_field], idx)
        eval_probes.append(
            {
                "text": normalize_text(identity_text, lowercase, max_length),
                "label": int(ex[sst_label_field]),
                "source": "eval_identity_probe",
            }
        )

    return {
        "d_opt": d_opt,
        "audit_disc": audit_disc_set,
        "audit_conf": audit_conf_set,
        "eval_sets": {"sst2": eval_sst, "yelp": eval_yelp, "imdb": eval_imdb},
        "eval_probes": eval_probes,
    }
