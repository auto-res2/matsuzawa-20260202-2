import hashlib
import math
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**31 - 1)


def opposite_sentiment(label: int) -> str:
    return "negative" if int(label) == 1 else "positive"


def label_to_int(label: str) -> int:
    label = label.lower()
    return 1 if "pos" in label or label.endswith("1") else 0


def has_negation(text: str) -> bool:
    text = text.lower()
    return (" not " in f" {text} ") or ("n't" in text)


def entity_rate_proxy(text: str) -> float:
    tokens = text.split()
    if not tokens:
        return 0.0
    caps = sum(1 for t in tokens if t[:1].isupper())
    return caps / max(1, len(tokens))


def has_identity_term(text: str) -> bool:
    text = text.lower()
    identity_terms = ["muslim", "black", "gay", "disabled"]
    return any(term in text for term in identity_terms)


def slice_key(text: str) -> str:
    length = len(text.split())
    length_bin = "short" if length <= 8 else ("med" if length <= 18 else "long")
    neg = int(has_negation(text))
    ent = "hiEnt" if entity_rate_proxy(text) >= 0.18 else "loEnt"
    ident = int(has_identity_term(text))
    return f"len_{length_bin}|neg_{neg}|{ent}|id_{ident}"


def fixed_slice_keys() -> List[str]:
    length_bins = ["short", "med", "long"]
    negs = [0, 1]
    ents = ["hiEnt", "loEnt"]
    ids = [0, 1]
    keys = []
    for lb in length_bins:
        for neg in negs:
            for ent in ents:
                for ident in ids:
                    keys.append(f"len_{lb}|neg_{neg}|{ent}|id_{ident}")
    return keys


class PromptRewriter:
    def __init__(self, name: str, device: torch.device, max_new_tokens: int, cache_dir: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
        try:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(name, cache_dir=cache_dir).to(device)
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(name, cache_dir=cache_dir).to(device)
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.is_encoder_decoder = bool(getattr(self.model.config, "is_encoder_decoder", False))
        self._ensure_pad_token()
        self.model.eval()

    def _ensure_pad_token(self) -> None:
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer pad_token_id could not be set.")
        if len(self.tokenizer) != self.model.get_input_embeddings().num_embeddings:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def build_input(self, prompt: str, text: str) -> str:
        return f"{prompt}\n\nText: {text}\nRewrite:"

    def generate(self, prompt: str, text: str, temperature: float = 0.7) -> str:
        input_text = self.build_input(prompt, text)
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)
        seed = stable_hash(prompt + text)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                generator=generator,
            )
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        if not self.is_encoder_decoder and decoded.startswith(input_text):
            decoded = decoded[len(input_text) :].strip()
        return decoded.split("\n")[0].strip()

    def propose_candidates(self, meta_prompt: str, k: int, temperature: float = 0.9) -> List[str]:
        inputs = self.tokenizer(
            meta_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)
        seed = stable_hash(meta_prompt)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=k,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                generator=generator,
            )
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        cleaned = []
        for text in decoded:
            prompt = text.strip().split("\n")[0]
            if 3 <= len(prompt.split()) <= 30:
                cleaned.append(prompt)
        return list(dict.fromkeys(cleaned))


class SentimentEvaluator:
    def __init__(self, name: str, device: torch.device, cache_dir: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.model = AutoModelForSequenceClassification.from_pretrained(name, cache_dir=cache_dir).to(device)
        if len(self.tokenizer) != self.model.get_input_embeddings().num_embeddings:
            self.model.resize_token_embeddings(len(self.tokenizer))
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Sentiment tokenizer pad_token_id could not be set.")
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()
        self.device = device

    def predict(self, texts: List[str]) -> List[str]:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            preds = logits.argmax(dim=-1).tolist()
        labels = []
        for idx in preds:
            label = self.model.config.id2label.get(idx, str(idx)).lower()
            if "pos" in label or label.endswith("1"):
                labels.append("positive")
            else:
                labels.append("negative")
        return labels


class SimilarityEncoder:
    def __init__(self, name: str, device: torch.device, cache_dir: str) -> None:
        self.model = SentenceTransformer(name, cache_folder=cache_dir, device=str(device))
        self.device = device

    def encode(self, texts: List[str]) -> torch.Tensor:
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        if isinstance(embeddings, torch.Tensor):
            return embeddings
        return torch.tensor(embeddings)


class FluencyEvaluator:
    def __init__(self, name: str, device: torch.device, cache_dir: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.model = AutoModelForCausalLM.from_pretrained(name, cache_dir=cache_dir).to(device)
        if len(self.tokenizer) != self.model.get_input_embeddings().num_embeddings:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()
        self.device = device

    def nll(self, text: str, max_length: int = 256) -> float:
        if not text.strip():
            return float("inf")
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(self.device)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        with torch.no_grad():
            out = self.model(**enc, labels=labels)
        return float(out.loss.item())


class ConstrainedEvaluator:
    def __init__(
        self,
        rewriter_name: str,
        sentiment_name: str,
        similarity_name: str,
        fluency_name: str,
        device: torch.device,
        max_new_tokens: int,
        cache_dir: str,
    ) -> None:
        self.device = device
        self.rewriter = PromptRewriter(rewriter_name, device, max_new_tokens, cache_dir)
        self.sentiment = SentimentEvaluator(sentiment_name, device, cache_dir)
        self.similarity = SimilarityEncoder(similarity_name, device, cache_dir)
        self.fluency = FluencyEvaluator(fluency_name, device, cache_dir)
        self.rewrite_cache: Dict[Tuple[str, str], str] = {}
        self.sent_cache: Dict[str, str] = {}
        self.embed_cache: Dict[str, torch.Tensor] = {}
        self.nll_cache: Dict[str, float] = {}
        self.success_cache: Dict[Tuple[str, str, int, float, float], Dict[str, Any]] = {}

    def clear_cache(self) -> None:
        self.rewrite_cache.clear()
        self.sent_cache.clear()
        self.embed_cache.clear()
        self.nll_cache.clear()
        self.success_cache.clear()

    def rewrite(self, prompt: str, text: str) -> str:
        key = (prompt, text)
        if key in self.rewrite_cache:
            return self.rewrite_cache[key]
        rewrite = self.rewriter.generate(prompt, text)
        self.rewrite_cache[key] = rewrite
        return rewrite

    def sentiment_label(self, text: str) -> str:
        if text in self.sent_cache:
            return self.sent_cache[text]
        label = self.sentiment.predict([text])[0]
        self.sent_cache[text] = label
        return label

    def embedding(self, text: str) -> torch.Tensor:
        if text in self.embed_cache:
            return self.embed_cache[text]
        emb = self.similarity.encode([text])[0].detach().cpu()
        self.embed_cache[text] = emb
        return emb

    def fluency_nll(self, text: str) -> float:
        if text in self.nll_cache:
            return self.nll_cache[text]
        nll = self.fluency.nll(text)
        self.nll_cache[text] = nll
        return nll

    def constrained_success(
        self,
        prompt: str,
        text: str,
        label: int,
        tau: float,
        nll_max: float,
    ) -> Dict[str, Any]:
        key = (prompt, text, int(label), float(tau), float(nll_max))
        if key in self.success_cache:
            return self.success_cache[key]
        rewrite = self.rewrite(prompt, text)
        if not rewrite.strip():
            result = {
                "success": 0,
                "rewrite": rewrite,
                "sim": 0.0,
                "nll": float("inf"),
                "ppl": float("inf"),
                "ok_sent": False,
                "pred_label": "",
                "target_label": opposite_sentiment(int(label)),
            }
            self.success_cache[key] = result
            return result
        pred = self.sentiment_label(rewrite)
        target = opposite_sentiment(int(label))
        ok_sent = pred == target
        emb_x = self.embedding(text)
        emb_y = self.embedding(rewrite)
        sim = float(torch.dot(emb_x, emb_y).item())
        nll = self.fluency_nll(rewrite)
        ppl = float(math.exp(nll)) if math.isfinite(nll) else float("inf")
        ok_sim = sim >= tau
        ok_fluency = nll <= nll_max
        success = int(ok_sent and ok_sim and ok_fluency)
        result = {
            "success": success,
            "rewrite": rewrite,
            "sim": sim,
            "nll": nll,
            "ppl": ppl,
            "ok_sent": ok_sent,
            "pred_label": pred,
            "target_label": target,
        }
        self.success_cache[key] = result
        return result


class AuxiliarySuccessPredictor(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def e_value_mixture(
    diffs: List[int],
    margin: float,
    lambdas: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.7),
) -> float:
    if not diffs:
        return 0.0
    diffs_t = torch.tensor(diffs, dtype=torch.float64)
    lambdas_t = torch.tensor(lambdas, dtype=torch.float64)
    log_es = []
    for lam in lambdas_t:
        factors = 1.0 + lam * (diffs_t - float(margin))
        factors = torch.clamp(factors, min=1e-12)
        log_es.append(torch.log(factors).sum())
    log_es = torch.stack(log_es)
    max_log = log_es.max()
    stable = torch.exp(log_es - max_log)
    mixture = stable.mean() * torch.exp(max_log)
    return float(mixture.item())


def paired_diffs(
    prompt: str,
    candidate: str,
    examples: List[Dict[str, Any]],
    evaluator: ConstrainedEvaluator,
    tau: float,
    nll_max: float,
    max_examples: Optional[int] = None,
) -> Tuple[List[int], Dict[str, List[int]]]:
    diffs: List[int] = []
    by_slice: Dict[str, List[int]] = defaultdict(list)
    for idx, ex in enumerate(examples):
        if max_examples is not None and idx >= max_examples:
            break
        base = evaluator.constrained_success(prompt, ex["text"], ex["label"], tau, nll_max)["success"]
        cand = evaluator.constrained_success(candidate, ex["text"], ex["label"], tau, nll_max)["success"]
        diff = int(cand - base)
        diffs.append(diff)
        sk = slice_key(ex["text"])
        by_slice[sk].append(diff)
    return diffs, by_slice


def mine_top_slices(by_slice: Dict[str, List[int]], top_j: int, min_size: int) -> List[str]:
    stats = []
    for key, diffs in by_slice.items():
        if len(diffs) < min_size:
            continue
        stats.append((float(np.mean(diffs)), key, len(diffs)))
    stats.sort(key=lambda x: x[0])
    return [key for _, key, _ in stats[:top_j]]


def quick_failures(
    prompt: str,
    examples: List[Dict[str, Any]],
    evaluator: ConstrainedEvaluator,
    tau: float,
    nll_max: float,
    max_fails: int = 6,
) -> List[str]:
    failures = []
    for ex in examples[:40]:
        res = evaluator.constrained_success(prompt, ex["text"], ex["label"], tau, nll_max)
        if res["success"] == 0:
            failures.append(f"Text: {ex['text'][:140]}")
        if len(failures) >= max_fails:
            break
    return failures


def propose_prompt_candidates(
    rewriter: PromptRewriter,
    prompt: str,
    failures: List[str],
    k: int,
) -> List[str]:
    examples_str = "\n".join([f"- {f}" for f in failures[:6]]) if failures else "- None"
    meta = (
        "You are editing an instruction prompt for a rewriting model.\n"
        "Goal: rewrite the text to the opposite sentiment while preserving meaning, entities, and topic.\n"
        "Write fluent English. Avoid repetition, gibberish, and negation tricks.\n"
        f"Current prompt: {prompt}\n\nFailures:\n{examples_str}\n\n"
        "Propose ONE improved prompt, <=25 words. Return ONLY the prompt text."
    )
    return rewriter.propose_candidates(meta, k=k)


def accept_cfse_safe(
    prompt: str,
    candidate: str,
    opt_examples: List[Dict[str, Any]],
    a_disc: List[Dict[str, Any]],
    a_conf: List[Dict[str, Any]],
    evaluator: ConstrainedEvaluator,
    wealth: float,
    alpha_max_spend: float,
    reward_frac: float,
    tau: float,
    nll_max: float,
    delta: float,
    top_j: int,
    min_support: int,
    max_examples: Optional[int],
) -> Tuple[bool, float, Dict[str, Any]]:
    spend = min(alpha_max_spend, wealth / 4.0)
    if spend <= 1e-12:
        return False, wealth, {"fail_gate": "no_wealth"}
    threshold = 1.0 / spend

    d_opt, _ = paired_diffs(prompt, candidate, opt_examples, evaluator, tau, nll_max, max_examples)
    e_opt = e_value_mixture(d_opt, margin=0.0)
    if e_opt < threshold:
        return False, wealth - spend, {"fail_gate": "opt", "E_opt": e_opt, "thresh": threshold}

    _, by_slice_disc = paired_diffs(prompt, candidate, a_disc, evaluator, tau, nll_max, max_examples)
    top_slices = mine_top_slices(by_slice_disc, top_j, min_support)

    d_all, by_slice_conf = paired_diffs(prompt, candidate, a_conf, evaluator, tau, nll_max, max_examples)
    e_all = e_value_mixture(d_all, margin=-delta)
    if e_all < threshold:
        return False, wealth - spend, {
            "fail_gate": "audit_all",
            "E_all": e_all,
            "thresh": threshold,
            "top_slices": top_slices,
        }

    min_slice_e = math.inf
    for sk in top_slices:
        diffs = by_slice_conf.get(sk, [])
        if len(diffs) < min_support:
            continue
        e_slice = e_value_mixture(diffs, margin=-delta)
        min_slice_e = min(min_slice_e, e_slice)
        if e_slice < threshold:
            return False, wealth - spend, {
                "fail_gate": f"slice:{sk}",
                "E_slice": e_slice,
                "thresh": threshold,
                "top_slices": top_slices,
            }

    new_wealth = min(0.2, (wealth - spend) + reward_frac * spend)
    return True, new_wealth, {
        "E_opt": e_opt,
        "E_all": e_all,
        "min_slice_e": min_slice_e,
        "top_slices": top_slices,
        "thresh": threshold,
    }


def accept_cavaspo_single(
    prompt: str,
    candidate: str,
    opt_examples: List[Dict[str, Any]],
    a_disc: List[Dict[str, Any]],
    a_conf: List[Dict[str, Any]],
    evaluator: ConstrainedEvaluator,
    wealth: float,
    alpha_max_spend: float,
    reward_frac: float,
    tau: float,
    nll_max: float,
    delta: float,
    min_support: int,
    max_examples: Optional[int],
) -> Tuple[bool, float, Dict[str, Any]]:
    spend = min(alpha_max_spend, wealth / 4.0)
    if spend <= 1e-12:
        return False, wealth, {"fail_gate": "no_wealth"}
    threshold = 1.0 / spend

    d_opt, _ = paired_diffs(prompt, candidate, opt_examples, evaluator, tau, nll_max, max_examples)
    e_opt = e_value_mixture(d_opt, margin=0.0)
    if e_opt < threshold:
        return False, wealth - spend, {"fail_gate": "opt", "E_opt": e_opt, "thresh": threshold}

    _, by_slice_disc = paired_diffs(prompt, candidate, a_disc, evaluator, tau, nll_max, max_examples)
    worst_slice = None
    worst_mean = math.inf
    for sk, diffs in by_slice_disc.items():
        if len(diffs) < min_support:
            continue
        mean_diff = float(np.mean(diffs))
        if mean_diff < worst_mean:
            worst_mean = mean_diff
            worst_slice = sk

    d_all, by_slice_conf = paired_diffs(prompt, candidate, a_conf, evaluator, tau, nll_max, max_examples)
    e_all = e_value_mixture(d_all, margin=-delta)
    if e_all < threshold:
        return False, wealth - spend, {
            "fail_gate": "audit_all",
            "E_all": e_all,
            "thresh": threshold,
            "worst_slice": worst_slice,
        }

    min_slice_e = math.inf
    if worst_slice is not None:
        diffs = by_slice_conf.get(worst_slice, [])
        if len(diffs) >= min_support:
            e_slice = e_value_mixture(diffs, margin=-delta)
            min_slice_e = e_slice
            if e_slice < threshold:
                return False, wealth - spend, {
                    "fail_gate": f"slice:{worst_slice}",
                    "E_slice": e_slice,
                    "thresh": threshold,
                    "worst_slice": worst_slice,
                }

    new_wealth = min(0.2, (wealth - spend) + reward_frac * spend)
    return True, new_wealth, {
        "E_opt": e_opt,
        "E_all": e_all,
        "min_slice_e": min_slice_e,
        "thresh": threshold,
        "worst_slice": worst_slice,
    }


def accept_bonferroni(
    prompt: str,
    candidate: str,
    opt_examples: List[Dict[str, Any]],
    a_disc: List[Dict[str, Any]],
    a_conf: List[Dict[str, Any]],
    evaluator: ConstrainedEvaluator,
    wealth: float,
    alpha_max_spend: float,
    reward_frac: float,
    tau: float,
    nll_max: float,
    delta: float,
    num_slices: int,
    min_support: int,
    bonferroni_family: str,
    candidates_per_iter: int,
    max_examples: Optional[int],
) -> Tuple[bool, float, Dict[str, Any]]:
    spend = min(alpha_max_spend, wealth / 4.0)
    if spend <= 1e-12:
        return False, wealth, {"fail_gate": "no_wealth"}
    threshold = 1.0 / spend

    d_opt, _ = paired_diffs(prompt, candidate, opt_examples, evaluator, tau, nll_max, max_examples)
    e_opt = e_value_mixture(d_opt, margin=0.0)
    if e_opt < threshold:
        return False, wealth - spend, {"fail_gate": "opt", "E_opt": e_opt, "thresh": threshold}

    d_all, by_slice_conf = paired_diffs(prompt, candidate, a_conf, evaluator, tau, nll_max, max_examples)
    e_all = e_value_mixture(d_all, margin=-delta)
    if e_all < threshold:
        return False, wealth - spend, {
            "fail_gate": "audit_all",
            "E_all": e_all,
            "thresh": threshold,
        }

    _, by_slice_disc = paired_diffs(prompt, candidate, a_disc, evaluator, tau, nll_max, max_examples)
    support = {k: len(by_slice_disc.get(k, [])) for k in fixed_slice_keys()}
    sorted_slices = sorted(support.items(), key=lambda kv: kv[1], reverse=True)
    selected_slices = [k for k, v in sorted_slices if v >= min_support][:num_slices]
    family_size = max(1, len(selected_slices))
    if bonferroni_family == "candidates_x_slices":
        family_size *= max(1, candidates_per_iter)
    slice_threshold = family_size / spend

    min_slice_e = math.inf
    for sk in selected_slices:
        diffs = by_slice_conf.get(sk, [])
        if len(diffs) < min_support:
            continue
        e_slice = e_value_mixture(diffs, margin=-delta)
        min_slice_e = min(min_slice_e, e_slice)
        if e_slice < slice_threshold:
            return False, wealth - spend, {
                "fail_gate": f"slice:{sk}",
                "E_slice": e_slice,
                "thresh": slice_threshold,
            }

    new_wealth = min(0.2, (wealth - spend) + reward_frac * spend)
    return True, new_wealth, {
        "E_opt": e_opt,
        "E_all": e_all,
        "min_slice_e": min_slice_e,
        "thresh": threshold,
        "slice_thresh": slice_threshold,
    }


def accept_self_refine(
    prompt: str,
    candidate: str,
    opt_examples: List[Dict[str, Any]],
    evaluator: ConstrainedEvaluator,
    tau: float,
    nll_max: float,
    max_examples: Optional[int],
) -> Tuple[bool, float, Dict[str, Any]]:
    d_opt, _ = paired_diffs(prompt, candidate, opt_examples, evaluator, tau, nll_max, max_examples)
    mean_diff = float(np.mean(d_opt)) if d_opt else 0.0
    if mean_diff > 0:
        return True, 0.0, {"mean_diff_opt": mean_diff}
    return False, 0.0, {"fail_gate": "opt", "mean_diff_opt": mean_diff}


def optimize_prompt(
    method_name: str,
    base_prompt: str,
    evaluator: ConstrainedEvaluator,
    d_opt: List[Dict[str, Any]],
    a_disc: List[Dict[str, Any]],
    a_conf: List[Dict[str, Any]],
    prompt_cfg: Any,
    log_fn: Optional[Callable[[Dict[str, Any], int], None]],
    global_step: int,
    max_gate_examples: Optional[int],
) -> Dict[str, Any]:
    prompt = base_prompt
    wealth = float(getattr(prompt_cfg, "alpha", 0.1))
    gate_fail_counts: Counter = Counter()
    accepted_steps = 0
    candidates_evaluated = 0
    tau = float(prompt_cfg.tau_similarity)
    nll_max = float(getattr(prompt_cfg, "nll_max", getattr(prompt_cfg, "pi_fluency_nll", 4.0)))

    method_lower = method_name.lower()
    if "fixed" in method_lower:
        return {
            "final_prompt": base_prompt,
            "gate_fail_counts": gate_fail_counts,
            "accepted_steps": 0,
            "candidates_evaluated": 0,
            "global_step": global_step,
        }

    if "self-refine" in method_lower or "self_refine" in method_lower or "vanilla" in method_lower:
        strategy = "self_refine"
    elif "cavaspo" in method_lower:
        strategy = "cavaspo"
    elif "bonferroni" in method_lower or "av-dr-spo" in method_lower:
        strategy = "bonferroni"
    elif getattr(prompt_cfg, "gatekeeping", "") == "bonferroni":
        strategy = "bonferroni"
    else:
        strategy = "cfse"

    for iter_idx in range(int(prompt_cfg.iterations)):
        failures = quick_failures(prompt, d_opt, evaluator, tau, nll_max)
        candidates = propose_prompt_candidates(
            evaluator.rewriter,
            prompt,
            failures,
            int(prompt_cfg.candidates_per_iter),
        )
        if not candidates:
            break
        accepted = False
        for cand_idx, candidate in enumerate(candidates):
            candidates_evaluated += 1
            if strategy == "bonferroni":
                ok, wealth, info = accept_bonferroni(
                    prompt,
                    candidate,
                    d_opt,
                    a_disc,
                    a_conf,
                    evaluator,
                    wealth,
                    float(prompt_cfg.alpha_max_spend),
                    float(prompt_cfg.reward_frac),
                    tau,
                    nll_max,
                    float(prompt_cfg.delta_noninferiority),
                    int(prompt_cfg.num_slices_tested),
                    int(prompt_cfg.slice_min_support),
                    str(prompt_cfg.bonferroni_family),
                    int(prompt_cfg.candidates_per_iter),
                    max_gate_examples,
                )
            elif strategy == "cavaspo":
                ok, wealth, info = accept_cavaspo_single(
                    prompt,
                    candidate,
                    d_opt,
                    a_disc,
                    a_conf,
                    evaluator,
                    wealth,
                    float(prompt_cfg.alpha_max_spend),
                    float(prompt_cfg.reward_frac),
                    tau,
                    nll_max,
                    float(prompt_cfg.delta_noninferiority),
                    int(prompt_cfg.slice_min_support),
                    max_gate_examples,
                )
            elif strategy == "self_refine":
                ok, _, info = accept_self_refine(
                    prompt,
                    candidate,
                    d_opt,
                    evaluator,
                    tau,
                    nll_max,
                    max_gate_examples,
                )
            else:
                ok, wealth, info = accept_cfse_safe(
                    prompt,
                    candidate,
                    d_opt,
                    a_disc,
                    a_conf,
                    evaluator,
                    wealth,
                    float(prompt_cfg.alpha_max_spend),
                    float(prompt_cfg.reward_frac),
                    tau,
                    nll_max,
                    float(prompt_cfg.delta_noninferiority),
                    int(prompt_cfg.J_top_slices),
                    int(prompt_cfg.slice_min_support),
                    max_gate_examples,
                )

            fail_gate = info.get("fail_gate")
            if not ok and fail_gate:
                gate_fail_counts[fail_gate] += 1

            if log_fn is not None:
                log_fn(
                    {
                        "train/iter": iter_idx,
                        "train/candidate_index": cand_idx,
                        "train/wealth": wealth,
                        "candidate/accepted": int(ok),
                        "candidate/fail_gate": fail_gate or "accepted",
                        "candidate/e_value_opt": info.get("E_opt", math.nan),
                        "candidate/e_value_all": info.get("E_all", math.nan),
                        "candidate/min_slice_e": info.get("min_slice_e", math.nan),
                        "candidate/mean_diff_opt": info.get("mean_diff_opt", math.nan),
                        "candidate/threshold": info.get("thresh", math.nan),
                    },
                    global_step,
                )
                global_step += 1

            if ok:
                prompt = candidate
                accepted_steps += 1
                accepted = True
                break
        if log_fn is not None:
            log_fn(
                {
                    "train/iter": iter_idx,
                    "train/accepted": int(accepted),
                    "train/wealth": wealth,
                    "train/accepted_steps": accepted_steps,
                    "train/candidates_evaluated": candidates_evaluated,
                },
                global_step,
            )
            global_step += 1
        if not accepted:
            break

    return {
        "final_prompt": prompt,
        "gate_fail_counts": gate_fail_counts,
        "accepted_steps": accepted_steps,
        "candidates_evaluated": candidates_evaluated,
        "global_step": global_step,
    }


def constrained_accuracy(
    prompt: str,
    examples: List[Dict[str, Any]],
    evaluator: ConstrainedEvaluator,
    tau: float,
    nll_max: float,
    return_confusion: bool = False,
    max_examples: Optional[int] = None,
):
    scores = []
    confusion = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for idx, ex in enumerate(examples):
        if max_examples is not None and idx >= max_examples:
            break
        res = evaluator.constrained_success(prompt, ex["text"], ex["label"], tau, nll_max)
        scores.append(res["success"])
        if return_confusion:
            pred = label_to_int(res["pred_label"])
            target = label_to_int(res["target_label"])
            if target == 1 and pred == 1:
                confusion["tp"] += 1
            elif target == 0 and pred == 0:
                confusion["tn"] += 1
            elif target == 0 and pred == 1:
                confusion["fp"] += 1
            else:
                confusion["fn"] += 1
    acc = float(np.mean(scores)) if scores else 0.0
    if return_confusion:
        return acc, confusion
    return acc


def evaluate_prompt(
    prompt: str,
    eval_sets: Dict[str, List[Dict[str, Any]]],
    evaluator: ConstrainedEvaluator,
    tau: float,
    nll_max: float,
    return_confusion: bool = False,
    max_examples: Optional[int] = None,
):
    metrics: Dict[str, float] = {}
    confusions: Dict[str, Dict[str, int]] = {}
    for name, examples in eval_sets.items():
        if return_confusion:
            acc, confusion = constrained_accuracy(
                prompt,
                examples,
                evaluator,
                tau,
                nll_max,
                return_confusion=True,
                max_examples=max_examples,
            )
            metrics[name] = acc
            confusions[name] = confusion
        else:
            metrics[name] = constrained_accuracy(
                prompt,
                examples,
                evaluator,
                tau,
                nll_max,
                max_examples=max_examples,
            )
    if return_confusion:
        return metrics, confusions
    return metrics


def flatten_eval_sets(eval_sets: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    combined = []
    for examples in eval_sets.values():
        combined.extend(examples)
    return combined


def worst_slice_accuracy(
    prompt: str,
    examples: List[Dict[str, Any]],
    evaluator: ConstrainedEvaluator,
    tau: float,
    nll_max: float,
    min_support: int,
    max_examples: Optional[int] = None,
) -> float:
    slice_scores: Dict[str, List[int]] = defaultdict(list)
    for idx, ex in enumerate(examples):
        if max_examples is not None and idx >= max_examples:
            break
        success = evaluator.constrained_success(prompt, ex["text"], ex["label"], tau, nll_max)["success"]
        sk = slice_key(ex["text"])
        slice_scores[sk].append(success)
    slice_accs = [np.mean(vals) for vals in slice_scores.values() if len(vals) >= min_support]
    if not slice_accs:
        return constrained_accuracy(prompt, examples, evaluator, tau, nll_max, max_examples=max_examples)
    return float(min(slice_accs))
