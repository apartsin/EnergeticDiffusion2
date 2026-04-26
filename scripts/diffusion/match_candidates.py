"""Look up reranked candidates against known molecules.

For each top SMILES from a rerank result file:
  1. Canonical SMILES.
  2. Internal lookup: exact match against the 382k training set
     (latents_expanded.pt) and against Tier-A/B subset.
  3. External: PubChem PUG-REST exact-SMILES → CID + IUPAC name (if any).
  4. External: NCI CACTUS → IUPAC name (different coverage).

Outputs a markdown table at <exp>/candidate_matches.md.

Rate-limited to be polite (~3 requests/s/service). Online steps fail
gracefully and continue.
"""
from __future__ import annotations
import argparse, json, re, sys, time, urllib.parse, urllib.request, urllib.error
from pathlib import Path

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
import torch

BASE = Path("E:/Projects/EnergeticDiffusion2")

UA = "EnergeticDiffusion2-research/1.0 (+local)"


def canon(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, canonical=True) if m else None


def http_get_json(url: str, timeout: int = 10) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        return json.loads(data) if data else None
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None
    except Exception:
        return None


def http_get_text(url: str, timeout: int = 10) -> str | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def pubchem_lookup(smi: str) -> dict:
    """Returns {'cid': int|None, 'name': str|None, 'iupac': str|None}."""
    out = {"cid": None, "name": None, "iupac": None}
    smi_enc = urllib.parse.quote(smi, safe="")
    url_cid = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smi_enc}/cids/JSON"
    js = http_get_json(url_cid)
    if not js:
        return out
    cids = (js.get("IdentifierList") or {}).get("CID") or []
    if not cids:
        return out
    out["cid"] = int(cids[0])
    # fetch IUPAC name
    url_name = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
                f"{out['cid']}/property/IUPACName/JSON")
    js = http_get_json(url_name)
    if js:
        try:
            out["iupac"] = js["PropertyTable"]["Properties"][0]["IUPACName"]
        except (KeyError, IndexError, TypeError):
            pass
    # fetch synonyms (first one is usually the common name)
    url_syn = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
                f"{out['cid']}/synonyms/JSON")
    js = http_get_json(url_syn)
    if js:
        try:
            syn = js["InformationList"]["Information"][0]["Synonym"]
            if syn:
                out["name"] = syn[0]
        except (KeyError, IndexError, TypeError):
            pass
    return out


def cactus_lookup(smi: str) -> str | None:
    smi_enc = urllib.parse.quote(smi, safe="")
    url = f"https://cactus.nci.nih.gov/chemical/structure/{smi_enc}/iupac_name"
    txt = http_get_text(url)
    if not txt: return None
    txt = txt.strip()
    if "<html" in txt.lower() or len(txt) > 400 or txt.lower().startswith("not found"):
        return None
    return txt


def parse_smiles_from_md(path: Path) -> list[str]:
    """Pulls SMILES strings from markdown tables (pattern  | `SMILES` | ) and
    bullet lists (- `SMILES`)."""
    if not path.exists(): return []
    text = path.read_text(encoding="utf-8")
    out = []
    for line in text.splitlines():
        m = re.findall(r"`([^`]+)`", line)
        for s in m:
            if any(c in s for c in "()[]=#") and not s.endswith(".md"):
                out.append(s)
    # de-dup, preserve order
    seen = set(); ded = []
    for s in out:
        if s not in seen:
            seen.add(s); ded.append(s)
    return ded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--base", default=str(BASE))
    ap.add_argument("--source", choices=["rerank", "rerank_multi", "both"],
                    default="both")
    ap.add_argument("--limit", type=int, default=20,
                    help="Max number of unique candidates to look up")
    ap.add_argument("--delay", type=float, default=0.4,
                    help="Seconds between external requests (rate-limit polite)")
    ap.add_argument("--no_pubchem", action="store_true")
    ap.add_argument("--no_cactus",  action="store_true")
    ap.add_argument("--with_feasibility", action="store_true",
                    help="Add SA + SC columns from sascorer + SCScorer")
    args = ap.parse_args()
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

    base = Path(args.base)
    exp  = Path(args.exp)
    if not exp.is_absolute(): exp = base / exp

    cand = []
    if args.source in ("rerank", "both"):
        cand += parse_smiles_from_md(exp / "rerank_results.md")
    if args.source in ("rerank_multi", "both"):
        cand += parse_smiles_from_md(exp / "rerank_multi.md")
    seen = set(); deduped = []
    for s in cand:
        c = canon(s)
        if c and c not in seen:
            seen.add(c); deduped.append(c)
    deduped = deduped[: args.limit]
    print(f"Looking up {len(deduped)} unique candidates")

    # internal reference set
    print("Loading internal training set …")
    blob = torch.load(base / "data/training/diffusion/latents_expanded.pt",
                       weights_only=False)
    cw = blob["cond_weight"]; cv = blob["cond_valid"]
    ref_canon = set()
    ref_canon_trusted = set()
    for i, smi in enumerate(blob["smiles"]):
        c = canon(smi)
        if not c: continue
        ref_canon.add(c)
        # trusted = at least one Tier-A/B label
        if bool((cw[i] >= 0.99).any()) and bool(cv[i].any()):
            ref_canon_trusted.add(c)
    print(f"  internal canonical SMILES: {len(ref_canon):,} total, "
          f"{len(ref_canon_trusted):,} Tier-A/B")

    feas_fns = None
    if args.with_feasibility:
        try:
            from feasibility_utils import real_sa, real_sc
            feas_fns = (real_sa, real_sc)
        except Exception as exc:
            print(f"feasibility disabled: {exc}")

    rows = []
    for k, smi in enumerate(deduped):
        rec = {
            "smi":            smi,
            "in_internal":    smi in ref_canon,
            "in_tier_ab":     smi in ref_canon_trusted,
            "pubchem_cid":    None,
            "pubchem_name":   None,
            "pubchem_iupac":  None,
            "cactus_iupac":   None,
            "sa":             None,
            "sc":             None,
        }
        if feas_fns is not None:
            rec["sa"] = feas_fns[0](smi)
            rec["sc"] = feas_fns[1](smi)
        print(f"[{k+1:>2}/{len(deduped)}] {smi[:60]}")
        if not args.no_pubchem:
            pc = pubchem_lookup(smi)
            rec.update({
                "pubchem_cid":   pc["cid"],
                "pubchem_name":  pc["name"],
                "pubchem_iupac": pc["iupac"],
            })
            if pc["cid"]:
                print(f"      pubchem CID {pc['cid']}  "
                      f"name={pc['name']!r}")
            else:
                print("      pubchem: no exact match")
            time.sleep(args.delay)
        if not args.no_cactus and not rec["pubchem_iupac"]:
            iup = cactus_lookup(smi)
            rec["cactus_iupac"] = iup
            if iup:
                print(f"      cactus iupac: {iup}")
            time.sleep(args.delay)
        rows.append(rec)

    # markdown table
    sa_hdr = " SA |" if args.with_feasibility else ""
    sc_hdr = " SC |" if args.with_feasibility else ""
    md = ["# Candidate-match report",
          f"Source: `{args.source}` from `{exp.name}`", "",
          f"Looked up {len(rows)} candidates.", "",
          f"| # | SMILES | in train | in TierA/B | PubChem CID | "
          f"Name (PubChem) | IUPAC (PubChem / CACTUS) |{sa_hdr}{sc_hdr}",
          "|" + "|".join(["---"] * (7 + (2 if args.with_feasibility else 0))) + "|"]
    name_cell = lambda r: r["pubchem_name"] or "—"
    iupac_cell = lambda r: r["pubchem_iupac"] or r["cactus_iupac"] or "—"
    cid_cell = lambda r: (
        f"[{r['pubchem_cid']}](https://pubchem.ncbi.nlm.nih.gov/compound/{r['pubchem_cid']})"
        if r["pubchem_cid"] else "—")
    fmt_score = lambda v: ("—" if v is None or (isinstance(v, float) and v != v)
                            else f"{v:.2f}")
    for k, r in enumerate(rows):
        sa_cell = f" {fmt_score(r['sa'])} |" if args.with_feasibility else ""
        sc_cell = f" {fmt_score(r['sc'])} |" if args.with_feasibility else ""
        md.append(f"| {k+1} | `{r['smi']}` | "
                  f"{'✓' if r['in_internal'] else '·'} | "
                  f"{'✓' if r['in_tier_ab'] else '·'} | "
                  f"{cid_cell(r)} | {name_cell(r)} | {iupac_cell(r)} |"
                  f"{sa_cell}{sc_cell}")
    md += ["", "Legend: ✓ = match in our database; PubChem CID links open "
           "the compound page."]

    n_known_pc  = sum(1 for r in rows if r["pubchem_cid"])
    n_known_int = sum(1 for r in rows if r["in_internal"])
    n_known_tab = sum(1 for r in rows if r["in_tier_ab"])
    md += ["", "## Summary",
           f"- Found in PubChem (exact SMILES): **{n_known_pc} / {len(rows)}**",
           f"- Found in our 382k training set: **{n_known_int} / {len(rows)}**",
           f"- Found in our Tier-A/B set: **{n_known_tab} / {len(rows)}**",
           "",
           f"Genuinely novel (none of the above): "
           f"**{sum(1 for r in rows if not (r['pubchem_cid'] or r['in_internal'] or r['in_tier_ab']))} / {len(rows)}**"]

    out = exp / "candidate_matches.md"
    out.write_text("\n".join(md), encoding="utf-8")
    json_out = exp / "candidate_matches.json"
    with open(json_out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved {out}")
    print(f"Saved {json_out}")


if __name__ == "__main__":
    sys.exit(main())
