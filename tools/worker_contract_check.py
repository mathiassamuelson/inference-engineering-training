#!/usr/bin/env python3
"""
worker_contract_check.py — deterministic conformance checker for RCA worker captures.

The worker tier emits a STRICT JSON contract (one object, no prose, no code fences) that the
orchestrator parses directly. Whether a completion honours that contract is a parser question,
not a judgement call — so it is measured here, deterministically, rather than by the LLM judge
(which is explicitly instructed to ignore formatting and would score this axis as noise).

Scores capture JSONs produced by tools/rca_quality_probe.py. For each probe completion it records:
  * strict_json      — the raw completion parsed as JSON with no cleanup (what the contract demands)
  * parseable        — parsed after fence-stripping / outermost-object extraction (lenient fallback)
  * needed_fence_strip — parseable only after cleanup (a format violation, but not a content failure)
  * truncated        — finish_reason == "length" (benign cause of invalid JSON; bump --max-tokens)
  * schema_problems  — contract-schema violations on the parsed object (empty == valid)
  * conformant       — strict_json AND schema-valid (the headline)

Design (matches the rest of the toolchain):
  * Model identity is read from the capture's `model` field — never hardcoded — and propagated into
    the report metadata, console header, and default output filename (so runs never silently clobber).
  * Provenance of the input capture (model, git_sha, git_dirty, system_prompt_sha256) is carried
    through; the tool's own git SHA is recorded too.
  * --expected-component is optional with no default (no hardcoded component); when given it both
    flags mismatches and tags the default output filename.

Exit code: 0 if every probe is strict-conformant, else 1 (the report is always written regardless).

Example:
    python3 tools/worker_contract_check.py \
        phase-3-optimization-and-quantization/week-13/results/exp_quality_rca_payment-service_gemma-4-12B-it-qat-w4a16-ct_<ts>.json \
        --expected-component payment-service \
        --results-dir phase-3-optimization-and-quantization/week-13/results
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
CONFIDENCE_VALUES = {"high", "medium", "low"}
TOP_LEVEL_KEYS = {"component", "in_scope", "findings", "out_of_scope_observations", "summary"}
FINDING_KEYS = {"signal", "evidence", "confidence"}


def _git(args: list[str]) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def git_provenance() -> dict[str, Any]:
    sha = _git(["rev-parse", "HEAD"])
    return {"tool_git_sha": sha or None, "tool_git_dirty": bool(_git(["status", "--porcelain"]))}


def utc_stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "-", name.split("/")[-1])


def parse_completion(text: str | None) -> dict[str, Any]:
    """Strict-first parse with a lenient fallback. Records how much cleanup was needed."""
    out: dict[str, Any] = {"strict_json": False, "parseable": False,
                           "needed_fence_strip": False, "obj": None, "parse_error": None}
    raw = (text or "").strip()
    if not raw:
        out["parse_error"] = "empty completion"
        return out
    try:  # strict: raw parses as-is
        out["obj"] = json.loads(raw)
        out["strict_json"] = True
        out["parseable"] = True
        return out
    except json.JSONDecodeError as exc:
        strict_err = str(exc)
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()  # lenient: drop fences
    for candidate in (cleaned, _outermost_object(cleaned)):
        if candidate is None:
            continue
        try:
            out["obj"] = json.loads(candidate)
            out["parseable"] = True
            out["needed_fence_strip"] = True
            out["parse_error"] = f"not strict ({strict_err}); parsed after cleanup"
            return out
        except json.JSONDecodeError:
            continue
    out["parse_error"] = f"unparseable: {strict_err}"
    return out


def _outermost_object(text: str) -> str | None:
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else None


def validate_contract(obj: Any, expected_component: str | None) -> list[str]:
    problems: list[str] = []
    if not isinstance(obj, dict):
        return ["top-level value is not a JSON object"]

    comp = obj.get("component")
    if not isinstance(comp, str) or not comp:
        problems.append("missing/empty 'component' (string)")
    elif expected_component and comp != expected_component:
        problems.append(f"component '{comp}' != expected '{expected_component}'")

    if not isinstance(obj.get("in_scope"), bool):
        problems.append("missing/non-bool 'in_scope'")

    findings = obj.get("findings")
    if not isinstance(findings, list):
        problems.append("missing/non-list 'findings'")
    else:
        for i, f in enumerate(findings):
            if not isinstance(f, dict):
                problems.append(f"findings[{i}] is not an object")
                continue
            if not isinstance(f.get("signal"), str) or not f.get("signal"):
                problems.append(f"findings[{i}].signal missing/empty")
            ev = f.get("evidence")
            if not isinstance(ev, list) or not all(isinstance(x, str) for x in ev):
                problems.append(f"findings[{i}].evidence must be a list of strings")
            if f.get("confidence") not in CONFIDENCE_VALUES:
                problems.append(f"findings[{i}].confidence must be one of {sorted(CONFIDENCE_VALUES)}")
            extra_f = set(f.keys()) - FINDING_KEYS
            if extra_f:
                problems.append(f"findings[{i}] unexpected key(s): {sorted(extra_f)}")

    oos = obj.get("out_of_scope_observations")
    if not isinstance(oos, list) or not all(isinstance(x, str) for x in oos):
        problems.append("'out_of_scope_observations' must be a list of strings")

    if not isinstance(obj.get("summary"), str) or not obj.get("summary"):
        problems.append("missing/empty 'summary' (string)")

    extra = set(obj.keys()) - TOP_LEVEL_KEYS
    if extra:
        problems.append(f"unexpected top-level key(s): {sorted(extra)}")
    return problems


def check_capture(capture: dict[str, Any], expected_component: str | None) -> dict[str, Any]:
    per_probe: list[dict[str, Any]] = []
    for r in capture.get("results", []):
        rid = r.get("id")
        finish = r.get("finish_reason")
        truncated = finish == "length"
        if "error" in r and "completion" not in r:
            per_probe.append({"id": rid, "title": r.get("title", ""), "finish_reason": finish,
                              "truncated": truncated, "strict_json": False, "parseable": False,
                              "needed_fence_strip": False, "schema_problems": ["probe capture failed"],
                              "conformant": False, "parse_error": r.get("error")})
            continue
        p = parse_completion(r.get("completion"))
        problems = validate_contract(p["obj"], expected_component) if p["parseable"] else []
        conformant = p["strict_json"] and not problems
        per_probe.append({"id": rid, "title": r.get("title", ""), "finish_reason": finish,
                          "truncated": truncated, "strict_json": p["strict_json"],
                          "parseable": p["parseable"], "needed_fence_strip": p["needed_fence_strip"],
                          "schema_problems": problems, "conformant": conformant,
                          "parse_error": p["parse_error"]})

    n = len(per_probe)
    strict = sum(x["strict_json"] for x in per_probe)
    parseable = sum(x["parseable"] for x in per_probe)
    schema_valid = sum(x["parseable"] and not x["schema_problems"] for x in per_probe)
    conformant = sum(x["conformant"] for x in per_probe)
    truncated = sum(x["truncated"] for x in per_probe)

    def rate(x: int) -> float:
        return round(x / n, 3) if n else 0.0

    summary = {
        "probes": n,
        "strict_json": strict, "parseable": parseable, "schema_valid": schema_valid,
        "strict_conformant": conformant, "truncated": truncated,
        "conformance_rate": rate(conformant), "strict_json_rate": rate(strict),
    }
    return {"summary": summary, "per_probe": per_probe}


def main() -> int:
    ap = argparse.ArgumentParser(description="Worker-contract conformance checker for RCA captures.")
    ap.add_argument("capture", help="Capture JSON produced by rca_quality_probe.py.")
    ap.add_argument("--expected-component", default=None,
                    help="If set, flag completions whose 'component' differs, and tag the output name.")
    ap.add_argument("--results-dir", default=".", help="Directory for the report JSON (created if missing).")
    ap.add_argument("--output", default=None, help="Explicit output path (default embeds model + component).")
    args = ap.parse_args()

    cap_path = Path(args.capture)
    if not cap_path.is_file():
        print(f"ERROR: capture not found: {cap_path}", file=sys.stderr)
        return 2
    capture = json.loads(cap_path.read_text(encoding="utf-8"))
    if "results" not in capture or "model" not in capture:
        print(f"ERROR: {cap_path} is not a recognized capture (missing 'results'/'model').", file=sys.stderr)
        return 2

    model = capture["model"]
    prov = git_provenance()
    ts = utc_stamp()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = Path(args.output)
    else:
        parts = ["worker_contract_check"]
        if args.expected_component:
            parts.append(sanitize(args.expected_component))
        parts.append(sanitize(model))
        out_path = results_dir / f"{'_'.join(parts)}_{ts}.json"

    scored = check_capture(capture, args.expected_component)
    s = scored["summary"]

    print("=" * 72)
    print("  Worker-Contract Conformance Check")
    print("=" * 72)
    print(f"  model        : {model}")
    print(f"  capture      : {cap_path}")
    print(f"  expected comp: {args.expected_component or '(not checked)'}")
    print(f"  tool git_sha : {prov['tool_git_sha']}  dirty={prov['tool_git_dirty']}")
    print("=" * 72)
    for p in scored["per_probe"]:
        flags = []
        if p["truncated"]:
            flags.append("TRUNCATED")
        if p["parseable"] and not p["strict_json"]:
            flags.append("needed-cleanup")
        tag = "ok" if p["conformant"] else "FAIL"
        extra = (" [" + ", ".join(flags) + "]") if flags else ""
        print(f"  {tag:<4} {str(p['id']):<32}{extra}")
        for prob in p["schema_problems"]:
            print(f"         - {prob}")

    out = {
        "schema_version": SCHEMA_VERSION,
        "tool": "worker_contract_check.py",
        "timestamp_utc": ts,
        "capture_path": str(cap_path),
        "model": model,
        "expected_component": args.expected_component,
        "capture_provenance": {
            "model": capture.get("model"),
            "git_sha": capture.get("git_sha"),
            "git_dirty": capture.get("git_dirty"),
            "system_prompt_sha256": capture.get("system_prompt_sha256"),
            "schema_version": capture.get("schema_version"),
            "timestamp_utc": capture.get("timestamp_utc"),
        },
        **prov,
        "summary": s,
        "per_probe": scored["per_probe"],
    }
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 72)
    print(f"  strict-conformant: {s['strict_conformant']}/{s['probes']}  "
          f"(strict_json {s['strict_json']}, schema_valid {s['schema_valid']}, "
          f"truncated {s['truncated']})")
    print(f"  wrote {out_path}")
    return 0 if s["strict_conformant"] == s["probes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
