import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from openai import OpenAI


# -----------------------------
# File helpers
# -----------------------------

def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def safe_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", s)


# -----------------------------
# Parsing helpers
# -----------------------------

def parse_reformulations(text: str, expected: int) -> List[str]:
    """
    Parse reframing output into a list of reformulations.
    Accepts 1. / 1) formatting; falls back to paragraph splits if needed.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    items: List[str] = []
    current = ""

    for ln in lines:
        if re.match(r"^\d+[\.\)]\s+", ln):
            if current:
                items.append(current.strip())
            current = re.sub(r"^\d+[\.\)]\s+", "", ln).strip()
        else:
            if current:
                current += " " + ln
            else:
                current = ln

    if current:
        items.append(current.strip())

    if len(items) < 2:
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        items = chunks

    return items[:expected]


def extract_json_object(text: str) -> Dict[str, Any]:
    """
    Extract a JSON object from judge output, robustly.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in judge output.")
    return json.loads(m.group(0).strip())


# -----------------------------
# OpenAI call wrapper (Responses API)
# -----------------------------

def llm_call(client: OpenAI, model: str, prompt: str, temperature: float, top_p: float, max_tokens: int) -> str:
    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )
    return resp.output_text


# -----------------------------
# Domain picker (non-repeating)
# -----------------------------

def pick_nonrepeating_domain(
    client: OpenAI,
    cfg: Dict[str, Any],
    domain_picker_tpl: str,
    reformulated_prompt: str,
    used_domains: set
) -> str:
    model = cfg["model"]["name"]

    prompt = (
        domain_picker_tpl
        .replace("[USED_DOMAINS]", ", ".join(sorted(used_domains)) if used_domains else "none")
        .replace("[REFORMULATED_PROMPT]", reformulated_prompt)
    )

    raw = client.responses.create(
        model=model,
        input=prompt,
        temperature=0.7,
        max_output_tokens=20
    ).output_text.strip()

    domain = raw.splitlines()[0].strip(" \"'.,;:()[]{}")

    # Retry if repeated
    retries = 0
    while domain.lower() in {d.lower() for d in used_domains} and retries < 5:
        raw = client.responses.create(
            model=model,
            input=prompt,
            temperature=0.9,
            max_output_tokens=20
        ).output_text.strip()
        domain = raw.splitlines()[0].strip(" \"'.,;:()[]{}")
        retries += 1

    used_domains.add(domain)
    return domain


# -----------------------------
# Judge + scoring (TWO-STAGE)
# -----------------------------

def judge_candidate(
    client: OpenAI,
    cfg: Dict[str, Any],
    tpls: Dict[str, str],
    original_prompt: str,
    candidate_text: str
) -> Dict[str, Any]:
    """
    Two-stage evaluation:
    1. Constraint satisfaction (filter garbage)
    2. Breakthrough potential (measure innovation)
    """
    model = cfg["model"]["name"]
    max_toks = cfg["model"]["max_tokens"]
    t = cfg["sampling"]["judge"]["temperature"]
    top_p = cfg["sampling"]["judge"]["top_p"]

    # STAGE 1: Constraint satisfaction (existing judge)
    prompt = (
        tpls["judge"]
        .replace("[ORIGINAL_PROMPT]", original_prompt)
        .replace("[CANDIDATE_RESPONSE]", candidate_text)
    )
    raw = llm_call(client, model, prompt, t, top_p, max_toks)
    constraint_result = extract_json_object(raw)

    cs = clamp01(float(constraint_result.get("constraint_satisfaction", 0.0)))
    us = clamp01(float(constraint_result.get("usefulness", 0.0)))
    
    discard_thresh = float(cfg["selection"]["discard_if_constraint_below"])
    discard = bool(constraint_result.get("discard", False)) or (cs < discard_thresh)

    # If discarded, skip breakthrough evaluation
    if discard:
        return {
            "constraint_satisfaction": cs,
            "usefulness": us,
            "constraint_overall": cs * 0.5 + us * 0.5,
            "breakthrough_scores": None,
            "breakthrough_potential": 0.0,
            "discard": True,
            "final_score": 0.0,
            "raw_constraint_judge": raw,
            "raw_breakthrough_judge": None
        }

    # STAGE 2: Breakthrough evaluation
    breakthrough_prompt = (
        tpls["breakthrough_judge"]
        .replace("[ORIGINAL_PROMPT]", original_prompt)
        .replace("[CANDIDATE_RESPONSE]", candidate_text)
    )
    breakthrough_raw = llm_call(client, model, breakthrough_prompt, t, top_p, max_toks)
    breakthrough_result = extract_json_object(breakthrough_raw)

    bp = clamp01(float(breakthrough_result.get("breakthrough_potential", 0.0)))
    
    # Compute final score
    weights = cfg["selection"]
    final_score = (
        weights["constraint_weight"] * cs +
        weights["usefulness_weight"] * us +
        weights["breakthrough_weight"] * bp
    )

    return {
        "constraint_satisfaction": cs,
        "usefulness": us,
        "constraint_overall": cs * 0.5 + us * 0.5,
        "breakthrough_scores": breakthrough_result,
        "breakthrough_potential": bp,
        "discard": False,
        "final_score": final_score,
        "raw_constraint_judge": raw,
        "raw_breakthrough_judge": breakthrough_raw
    }


def select_top_k(candidates: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """
    Keep top-k by final_score among non-discarded if possible; otherwise among all.
    """
    non_discarded = [c for c in candidates if not c["judge"]["discard"]]
    pool = non_discarded if non_discarded else candidates
    pool_sorted = sorted(pool, key=lambda x: x["judge"]["final_score"], reverse=True)
    return pool_sorted[:k]


# -----------------------------
# Candidate generators
# -----------------------------

def generate_single_pass_candidates(
    client: OpenAI,
    cfg: Dict[str, Any],
    original_prompt: str,
    n: int
) -> List[Dict[str, Any]]:
    """
    Baseline: N independent samples directly from original prompt.
    """
    model = cfg["model"]["name"]
    max_toks = cfg["model"]["max_tokens"]
    t = cfg["sampling"]["single_pass"]["temperature"]
    top_p = cfg["sampling"]["single_pass"]["top_p"]

    out: List[Dict[str, Any]] = []
    for _ in range(n):
        txt = llm_call(client, model, original_prompt, t, top_p, max_toks)
        out.append({"mode": "single_pass", "reformulation": "", "text": txt})
    return out


def generate_reformulations(
    client: OpenAI,
    cfg: Dict[str, Any],
    reframing_tpl: str,
    original_prompt: str
) -> List[str]:
    model = cfg["model"]["name"]
    max_toks = cfg["model"]["max_tokens"]
    t = cfg["sampling"]["reframing"]["temperature"]
    top_p = cfg["sampling"]["reframing"]["top_p"]
    k = int(cfg["sampling"]["reframing"]["num_reformulations"])

    prompt = reframing_tpl.replace("[ORIGINAL_PROMPT]", original_prompt)
    txt = llm_call(client, model, prompt, t, top_p, max_toks)
    return parse_reformulations(txt, expected=k)


def generate_structured_candidates(
    client: OpenAI,
    cfg: Dict[str, Any],
    tpls: Dict[str, str],
    original_prompt: str
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Divergent–convergent candidate generation with strict budget:
      3 reformulations
      per reformulation:
        3 structural_transfer (non-repeating domains) + 3 constraint_violation
      => 18 total candidates
    """
    model = cfg["model"]["name"]
    max_toks = cfg["model"]["max_tokens"]

    reformulations = generate_reformulations(client, cfg, tpls["reframing"], original_prompt)

    comb_n = int(cfg["sampling"]["combinational"]["candidates_per_reformulation"])
    trans_n = int(cfg["sampling"]["transformational"]["candidates_per_reformulation"])

    comb_t = cfg["sampling"]["combinational"]["temperature"]
    comb_top_p = cfg["sampling"]["combinational"]["top_p"]

    trans_t = cfg["sampling"]["transformational"]["temperature"]
    trans_top_p = cfg["sampling"]["transformational"]["top_p"]

    candidates: List[Dict[str, Any]] = []

    used_domains = set()  # ensures no repeats within this prompt

    for p_i in reformulations:
        # structural transfer (combinational)
        for _ in range(comb_n):
            domain = pick_nonrepeating_domain(
                client=client,
                cfg=cfg,
                domain_picker_tpl=tpls["domain_picker"],
                reformulated_prompt=p_i,
                used_domains=used_domains
            )

            prompt = (
                tpls["combinational"]
                .replace("[REFORMULATED_PROMPT]", p_i)
                .replace("the provided unrelated domain", domain)
            )

            txt = llm_call(client, model, prompt, comb_t, comb_top_p, max_toks)
            candidates.append({
                "mode": "structural_transfer",
                "domain": domain,
                "reformulation": p_i,
                "text": txt
            })

        # constraint violation (transformational)
        for _ in range(trans_n):
            prompt = tpls["transformational"].replace("[REFORMULATED_PROMPT]", p_i)
            txt = llm_call(client, model, prompt, trans_t, trans_top_p, max_toks)
            candidates.append({
                "mode": "constraint_violation",
                "reformulation": p_i,
                "text": txt
            })

    return candidates, reformulations


# -----------------------------
# Run conditions
# -----------------------------

def run_single_pass_18_keep5(
    client: OpenAI,
    cfg: Dict[str, Any],
    tpls: Dict[str, str],
    original_prompt: str
) -> Dict[str, Any]:
    N = int(cfg["budget"]["max_candidates_per_prompt"])  # 18
    K = int(cfg["budget"]["keep_top_k"])                 # 5

    raw_cands = generate_single_pass_candidates(client, cfg, original_prompt, N)

    judged: List[Dict[str, Any]] = []
    for c in raw_cands:
        j = judge_candidate(client, cfg, tpls, original_prompt, c["text"])
        judged.append({**c, "judge": j})

    topk = select_top_k(judged, K)
    best = topk[0]

    return {
        "reformulations": [],
        "candidates": judged,
        "top_k": topk,
        "final_output": best["text"],
        "selected": {
            "mode": best["mode"],
            "reformulation": best["reformulation"],
            "scores": best["judge"]
        }
    }


def run_divergent_convergent_18_keep5(
    client: OpenAI,
    cfg: Dict[str, Any],
    tpls: Dict[str, str],
    original_prompt: str
) -> Dict[str, Any]:
    K = int(cfg["budget"]["keep_top_k"])  # 5

    raw_cands, reformulations = generate_structured_candidates(client, cfg, tpls, original_prompt)

    judged: List[Dict[str, Any]] = []
    for c in raw_cands:
        j = judge_candidate(client, cfg, tpls, original_prompt, c["text"])
        judged.append({**c, "judge": j})

    topk = select_top_k(judged, K)
    best = topk[0]

    return {
        "reformulations": reformulations,
        "candidates": judged,
        "top_k": topk,
        "final_output": best["text"],
        "selected": {
            "mode": best["mode"],
            "reformulation": best["reformulation"],
            "scores": best["judge"]
        }
    }


# -----------------------------
# Main runner
# -----------------------------

def main():
    root = Path(__file__).parent
    cfg = load_json(root / "experiment_config.json")
    prompts = load_json(root / "prompts.json")["prompts"]

    tpls = {
        "reframing": load_text(root / "problem_reframing.txt"),
        "combinational": load_text(root / "combinational.txt"),
        "transformational": load_text(root / "transformational.txt"),
        "judge": load_text(root / "judge.txt"),
        "breakthrough_judge": load_text(root / "breakthrough_judge.txt"),
        "domain_picker": load_text(root / "domain_picker.txt"),
    }

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY='your_key'")

    client = OpenAI()

    runs_dir = root / "runs"
    ensure_dir(runs_dir)

    for pr in prompts:
        pid = pr["id"]
        ptext = pr["text"]

        prompt_dir = runs_dir / safe_filename(pid)
        ensure_dir(prompt_dir)

        for cond in cfg["conditions"]:
            out_path = prompt_dir / f"{cond}.json"
            if out_path.exists():
                print(f"[skip] {pid} / {cond} (already exists)")
                continue

            print(f"[run] {pid} / {cond}")

            record: Dict[str, Any] = {
                "timestamp": now_iso(),
                "prompt_id": pid,
                "prompt": ptext,
                "condition": cond,
                "config": cfg
            }

            if cond == "single_pass_18_keep5":
                record["result"] = run_single_pass_18_keep5(client, cfg, tpls, ptext)

            elif cond == "divergent_convergent_18_keep5":
                record["result"] = run_divergent_convergent_18_keep5(client, cfg, tpls, ptext)

            else:
                raise ValueError(f"Unknown condition: {cond}")

            out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            print(f"[saved] {out_path}")

    print("\nDone. Outputs are in ./runs/")


if __name__ == "__main__":
    main()