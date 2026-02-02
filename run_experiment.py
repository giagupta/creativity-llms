import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set
import sys

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


def extract_assumption(text: str) -> str:
    """
    Extract the assumption being broken from a constraint violation response.
    """
    # Look for patterns like "assumption:", "breaking:", "I am breaking", etc.
    patterns = [
        r"assumption[:\s]+([^.]+)",
        r"breaking[:\s]+([^.]+)",
        r"I am breaking[:\s]+([^.]+)",
        r"assumption broken[:\s]+([^.]+)",
    ]
    
    text_lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            assumption = match.group(1).strip()
            # Clean up and truncate to first 100 chars
            assumption = assumption.split('.')[0].strip()
            return assumption[:100]
    
    # Fallback: just take first 100 chars
    return text[:100].strip()


# -----------------------------
# OpenAI call wrapper with retry
# -----------------------------

def llm_call(client: OpenAI, model: str, prompt: str, temperature: float, 
             top_p: float, max_tokens: int, max_retries: int = 3) -> str:
    """Call OpenAI API with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=model,
                input=prompt,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_tokens,
            )
            return resp.output_text
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt
            print(f"  API error (attempt {attempt + 1}/{max_retries}): {e}")
            print(f"  Retrying in {wait_time}s...")
            time.sleep(wait_time)
    
    raise RuntimeError("Max retries exceeded")


# -----------------------------
# Domain picker (non-repeating)
# -----------------------------

def pick_nonrepeating_domain(
    client: OpenAI,
    cfg: Dict[str, Any],
    domain_picker_tpl: str,
    reformulated_prompt: str,
    used_domains: Set[str]
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

    # STAGE 1: Constraint satisfaction
    prompt = (
        tpls["judge"]
        .replace("[ORIGINAL_PROMPT]", original_prompt)
        .replace("[CANDIDATE_RESPONSE]", candidate_text)
    )
    
    try:
        raw = llm_call(client, model, prompt, t, top_p, max_toks)
        constraint_result = extract_json_object(raw)
    except Exception as e:
        print(f"  Warning: Constraint judge failed: {e}")
        return {
            "constraint_satisfaction": 0.0,
            "usefulness": 0.0,
            "constraint_overall": 0.0,
            "breakthrough_scores": None,
            "breakthrough_potential": 0.0,
            "discard": True,
            "final_score": 0.0,
            "raw_constraint_judge": str(e),
            "raw_breakthrough_judge": None
        }

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
    
    try:
        breakthrough_raw = llm_call(client, model, breakthrough_prompt, t, top_p, max_toks)
        breakthrough_result = extract_json_object(breakthrough_raw)
    except Exception as e:
        print(f"  Warning: Breakthrough judge failed: {e}")
        final_score = 0.5 * cs + 0.5 * us
        return {
            "constraint_satisfaction": cs,
            "usefulness": us,
            "constraint_overall": cs * 0.5 + us * 0.5,
            "breakthrough_scores": None,
            "breakthrough_potential": 0.0,
            "discard": False,
            "final_score": final_score,
            "raw_constraint_judge": raw,
            "raw_breakthrough_judge": str(e)
        }

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
    WITH DIVERSITY ENFORCEMENT.
    """
    model = cfg["model"]["name"]
    max_toks = cfg["model"]["max_tokens"]
    t = cfg["sampling"]["single_pass"]["temperature"]
    top_p = cfg["sampling"]["single_pass"]["top_p"]

    # Add diversity instruction
    diversity_prompt = f"""{original_prompt}

CRITICAL: Generate a solution that is SUBSTANTIALLY DIFFERENT from typical or obvious approaches. Avoid the most common solutions that immediately come to mind."""

    out: List[Dict[str, Any]] = []
    for i in range(n):
        print(f"    Generating candidate {i+1}/{n}...", end='\r')
        try:
            txt = llm_call(client, model, diversity_prompt, t, top_p, max_toks)
            out.append({"mode": "single_pass", "reformulation": "", "text": txt})
        except Exception as e:
            print(f"\n    Warning: Failed to generate candidate {i+1}: {e}")
            out.append({"mode": "single_pass", "reformulation": "", 
                       "text": f"[Generation failed: {e}]"})
    
    print()
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
    Divergent–convergent candidate generation with diversity enforcement:
      3 reformulations
      per reformulation:
        3 structural_transfer (non-repeating domains) + 3 constraint_violation (non-repeating assumptions)
      => 18 total candidates
    """
    model = cfg["model"]["name"]
    max_toks = cfg["model"]["max_tokens"]

    print("  Generating reformulations...")
    reformulations = generate_reformulations(client, cfg, tpls["reframing"], original_prompt)
    print(f"  Generated {len(reformulations)} reformulations")

    comb_n = int(cfg["sampling"]["combinational"]["candidates_per_reformulation"])
    trans_n = int(cfg["sampling"]["transformational"]["candidates_per_reformulation"])

    comb_t = cfg["sampling"]["combinational"]["temperature"]
    comb_top_p = cfg["sampling"]["combinational"]["top_p"]

    trans_t = cfg["sampling"]["transformational"]["temperature"]
    trans_top_p = cfg["sampling"]["transformational"]["top_p"]

    candidates: List[Dict[str, Any]] = []
    used_domains: Set[str] = set()
    used_assumptions: Set[str] = set()

    total_expected = len(reformulations) * (comb_n + trans_n)
    current = 0

    for p_idx, p_i in enumerate(reformulations):
        print(f"  Reformulation {p_idx+1}/{len(reformulations)}")
        
        # Track domains used in THIS reformulation
        reformulation_domains: Set[str] = set()
        
        # structural transfer (combinational) with domain tracking
        for j in range(comb_n):
            current += 1
            print(f"    [{current}/{total_expected}] Structural transfer {j+1}/{comb_n}...", end='\r')
            
            try:
                domain = pick_nonrepeating_domain(
                    client=client,
                    cfg=cfg,
                    domain_picker_tpl=tpls["domain_picker"],
                    reformulated_prompt=p_i,
                    used_domains=used_domains
                )
                
                reformulation_domains.add(domain)

                # Use improved template with used domains list
                prompt = (
                    tpls["combinational"]
                    .replace("[REFORMULATED_PROMPT]", p_i)
                    .replace("[DOMAIN]", domain)
                    .replace("[USED_DOMAINS]", ", ".join(sorted(used_domains - {domain})) if used_domains - {domain} else "none")
                )

                txt = llm_call(client, model, prompt, comb_t, comb_top_p, max_toks)
                candidates.append({
                    "mode": "structural_transfer",
                    "domain": domain,
                    "reformulation": p_i,
                    "text": txt
                })
            except Exception as e:
                print(f"\n    Warning: Structural transfer failed: {e}")
                candidates.append({
                    "mode": "structural_transfer",
                    "domain": "unknown",
                    "reformulation": p_i,
                    "text": f"[Generation failed: {e}]"
                })

        # constraint violation (transformational) with assumption tracking
        for j in range(trans_n):
            current += 1
            print(f"    [{current}/{total_expected}] Constraint violation {j+1}/{trans_n}...", end='\r')
            
            try:
                # Use improved template with used assumptions list
                prompt = (
                    tpls["transformational"]
                    .replace("[REFORMULATED_PROMPT]", p_i)
                    .replace("[USED_ASSUMPTIONS]", ", ".join(sorted(used_assumptions)) if used_assumptions else "none")
                )
                
                txt = llm_call(client, model, prompt, trans_t, trans_top_p, max_toks)
                
                # Extract and track assumption
                assumption = extract_assumption(txt)
                used_assumptions.add(assumption)
                
                candidates.append({
                    "mode": "constraint_violation",
                    "assumption": assumption,
                    "reformulation": p_i,
                    "text": txt
                })
            except Exception as e:
                print(f"\n    Warning: Constraint violation failed: {e}")
                candidates.append({
                    "mode": "constraint_violation",
                    "assumption": "unknown",
                    "reformulation": p_i,
                    "text": f"[Generation failed: {e}]"
                })
    
    print()
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

    print(f"  Generating {N} single-pass candidates (with diversity enforcement)...")
    raw_cands = generate_single_pass_candidates(client, cfg, original_prompt, N)

    print(f"  Judging {len(raw_cands)} candidates...")
    judged: List[Dict[str, Any]] = []
    for i, c in enumerate(raw_cands):
        print(f"    Judging candidate {i+1}/{len(raw_cands)}...", end='\r')
        j = judge_candidate(client, cfg, tpls, original_prompt, c["text"])
        judged.append({**c, "judge": j})
    print()

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

    print(f"  Judging {len(raw_cands)} candidates...")
    judged: List[Dict[str, Any]] = []
    for i, c in enumerate(raw_cands):
        print(f"    Judging candidate {i+1}/{len(raw_cands)}...", end='\r')
        j = judge_candidate(client, cfg, tpls, original_prompt, c["text"])
        judged.append({**c, "judge": j})
    print()

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


def run_evolutionary_examples(
    client: OpenAI,
    cfg: Dict[str, Any],
    tpls: Dict[str, str],
    original_prompt: str
) -> Dict[str, Any]:
    """
    REVISED evolutionary approach:
    Generation 0: Diverse exploration (18 candidates)
    Generation 1: Use top-3 as EXAMPLES to generate 6 NEW diverse ideas
    """
    
    # GENERATION 0: Diverse exploration
    print("  GENERATION 0: Diverse exploration")
    gen0_cands, reformulations = generate_structured_candidates(client, cfg, tpls, original_prompt)

    print(f"  Judging {len(gen0_cands)} generation 0 candidates...")
    gen0_judged: List[Dict[str, Any]] = []
    for i, c in enumerate(gen0_cands):
        print(f"    Judging candidate {i+1}/{len(gen0_cands)}...", end='\r')
        j = judge_candidate(client, cfg, tpls, original_prompt, c["text"])
        gen0_judged.append({**c, "judge": j, "generation": 0})
    print()
    
    # Select top-3 as EXAMPLES
    top_3_examples = select_top_k(gen0_judged, k=3)
    
    print(f"  Top-3 examples selected:")
    for i, ex in enumerate(top_3_examples):
        print(f"    Example {i+1}: score={ex['judge']['final_score']:.3f}, "
              f"breakthrough={ex['judge']['breakthrough_potential']:.3f}")
    
    # GENERATION 1: Use examples to inspire NEW ideas
    print("\n  GENERATION 1: Example-inspired generation")
    gen1_cands: List[Dict[str, Any]] = []
    
    model = cfg["model"]["name"]
    max_toks = cfg["model"]["max_tokens"]
    
    # Create example text
    examples_text = ""
    for i, ex in enumerate(top_3_examples, 1):
        examples_text += f"\nEXAMPLE {i}:\n{ex['text']}\n"
    
    # Generate 6 new candidates inspired by examples
    inspiration_prompt = f"""Given this problem:
{original_prompt}

Here are three high-quality example solutions that take creative approaches:
{examples_text}

Your task: Generate a NEW solution that:
1. Is inspired by the creativity and depth shown in the examples
2. Explores a DIFFERENT angle or approach than any of the examples
3. Matches or exceeds their level of insight and innovation
4. Is concrete, feasible, and clearly justified

Do NOT refine or extend any example. Generate a genuinely NEW solution inspired by their quality.
(Maximum two paragraphs.)"""
    
    for i in range(6):
        print(f"    Generating example-inspired candidate {i+1}/6...", end='\r')
        
        try:
            # Use high temperature to ensure diversity
            new_text = llm_call(client, model, inspiration_prompt, 1.0, 1.0, max_toks)
            
            gen1_cands.append({
                "mode": "example_inspired",
                "inspiration_from": "top_3",
                "reformulation": "",
                "text": new_text,
                "generation": 1
            })
        except Exception as e:
            print(f"\n      Warning: Example-inspired generation failed: {e}")
            gen1_cands.append({
                "mode": "example_inspired",
                "inspiration_from": "top_3",
                "reformulation": "",
                "text": f"[Generation failed: {e}]",
                "generation": 1
            })
    
    print()
    print(f"  Judging {len(gen1_cands)} generation 1 candidates...")
    gen1_judged: List[Dict[str, Any]] = []
    for i, c in enumerate(gen1_cands):
        print(f"    Judging candidate {i+1}/{len(gen1_cands)}...", end='\r')
        j = judge_candidate(client, cfg, tpls, original_prompt, c["text"])
        gen1_judged.append({**c, "judge": j})
    print()
    
    # FINAL SELECTION: Combine both generations
    print("  Final selection with elitism...")
    all_candidates = gen0_judged + gen1_judged
    
    K = int(cfg["budget"]["keep_top_k"])
    final_top_k = select_top_k(all_candidates, k=K)
    best = final_top_k[0]
    
    # Compute statistics
    gen0_best_score = max(c["judge"]["final_score"] for c in gen0_judged)
    gen1_best_score = max(c["judge"]["final_score"] for c in gen1_judged)
    
    gen1_better_count = sum(
        1 for c in gen1_judged 
        if c["judge"]["final_score"] > gen0_best_score
    )
    
    print(f"  Generation 0 best score: {gen0_best_score:.3f}")
    print(f"  Generation 1 best score: {gen1_best_score:.3f}")
    print(f"  Gen-1 exceeded gen-0 best: {gen1_better_count}/{len(gen1_judged)}")
    print(f"  Final selected: generation {best['generation']}, score {best['judge']['final_score']:.3f}")
    
    return {
        "reformulations": reformulations,
        "generation_0": gen0_judged,
        "generation_1": gen1_judged,
        "top_3_examples": top_3_examples,
        "candidates": all_candidates,
        "top_k": final_top_k,
        "final_output": best["text"],
        "selected": {
            "generation": best["generation"],
            "mode": best["mode"],
            "reformulation": best.get("reformulation", ""),
            "scores": best["judge"]
        },
        "improvement_stats": {
            "gen0_best_score": gen0_best_score,
            "gen1_best_score": gen1_best_score,
            "improvement": gen1_best_score - gen0_best_score,
            "gen1_better_count": gen1_better_count,
            "total_gen1": len(gen1_judged),
            "improvement_rate": gen1_better_count / len(gen1_judged) if gen1_judged else 0
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
        "combinational": load_text(root / "combinational.txt"),  # ← CHANGED
        "transformational": load_text(root / "transformational.txt"),  # ← CHANGED
        "judge": load_text(root / "judge.txt"),
        "breakthrough_judge": load_text(root / "breakthrough_judge.txt"),
        "domain_picker": load_text(root / "domain_picker.txt"),
    }

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY='your_key'")

    client = OpenAI()

    runs_dir = root / "runs"
    ensure_dir(runs_dir)

    # Count total work
    total_runs = len(prompts) * len(cfg["conditions"])
    completed = 0
    
    print(f"\n{'='*80}")
    print(f"STARTING EXPERIMENT: {total_runs} total runs")
    print(f"{'='*80}\n")

    for pr_idx, pr in enumerate(prompts):
        pid = pr["id"]
        ptext = pr["text"]

        prompt_dir = runs_dir / safe_filename(pid)
        ensure_dir(prompt_dir)

        print(f"\n[PROMPT {pr_idx+1}/{len(prompts)}] {pid}")
        print(f"{'='*80}")

        for cond in cfg["conditions"]:
            out_path = prompt_dir / f"{cond}.json"
            
            if out_path.exists():
                print(f"[SKIP] {cond} (already exists)")
                completed += 1
                continue

            print(f"\n[RUN] {cond}")
            print(f"Progress: {completed}/{total_runs} completed")
            
            start_time = time.time()

            try:
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
                
                elif cond == "evolutionary_examples":
                    record["result"] = run_evolutionary_examples(client, cfg, tpls, ptext)

                else:
                    raise ValueError(f"Unknown condition: {cond}")

                elapsed = time.time() - start_time
                record["elapsed_seconds"] = elapsed

                out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
                
                print(f"\n[SAVED] {out_path}")
                print(f"Time: {elapsed:.1f}s")
                print(f"Selected candidate breakthrough: {record['result']['selected']['scores']['breakthrough_potential']:.3f}")
                
                completed += 1

            except KeyboardInterrupt:
                print("\n\n[INTERRUPTED] Saving progress and exiting...")
                print(f"Completed {completed}/{total_runs} runs")
                sys.exit(0)
                
            except Exception as e:
                print(f"\n[ERROR] Failed to run {pid} / {cond}: {e}")
                print(f"Continuing with next condition...")
                import traceback
                traceback.print_exc()

    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Completed: {completed}/{total_runs} runs")
    print(f"Results saved to: {runs_dir}/")
    print(f"{'='*80}\n")
    print("\nRun 'python analyze_results.py' to generate analysis and visualizations.")


if __name__ == "__main__":
    main()