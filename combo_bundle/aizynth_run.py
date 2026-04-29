"""R1.b AiZynthFinder retrosynthesis on top-3 leads.

Downloads AiZynthFinder USPTO public model + runs MCTS-based retrosynthetic
search for each target. Reports for each lead: number of solutions found,
top-ranked route score, number of reactions, and the route tree as JSON.

Note: AiZynthFinder's USPTO template set is drug-discovery-biased and may
miss energetic-specific reactions (nitration via mixed acid, N2O5, KMnO4
oxidation, etc.). The result is a plausibility lower-bound: if a route is
found, the target is reachable; if no route is found within the search
budget, the target may still be synthesisable via energetics-specific
methods not in the USPTO template set.
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="smiles_targets.json")
    ap.add_argument("--config", default="config.yml")
    ap.add_argument("--out", default="results/aizynth_results.json")
    ap.add_argument("--max_iterations", type=int, default=200,
                    help="MCTS budget per target")
    ap.add_argument("--time_limit", type=int, default=300,
                    help="seconds per target before abort")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print(f"[train] Loading targets from {args.targets}"); sys.stdout.flush()
    cfg = json.loads(Path(args.targets).read_text())
    targets = cfg["leads"]
    print(f"[train] {len(targets)} target compounds"); sys.stdout.flush()

    # Initialise AiZynthFinder
    print(f"[train] Loading AiZynthFinder ..."); sys.stdout.flush()
    from aizynthfinder.aizynthfinder import AiZynthFinder
    finder = AiZynthFinder(configfile=args.config)
    finder.stock.select(["zinc"])
    finder.expansion_policy.select(["uspto"])
    finder.filter_policy.select(["uspto"])
    print(f"[train] Stock: zinc, Expansion: uspto, Filter: uspto"); sys.stdout.flush()

    # Search budget
    finder.config.search.iteration_limit = args.max_iterations
    finder.config.search.time_limit = args.time_limit

    results = []
    for i, t in enumerate(targets, 1):
        print(f"\n[train] {i}/{len(targets)} loss=0.0000 target={t['id']}")
        print(f"  SMILES: {t['smiles']}")
        sys.stdout.flush()
        result = {"id": t["id"], "smiles": t["smiles"], "name": t.get("name", "")}
        try:
            t0 = time.time()
            finder.target_smiles = t["smiles"]
            finder.tree_search()
            finder.build_routes()
            elapsed = time.time() - t0
            n_routes = len(finder.routes)
            result["n_routes_found"] = n_routes
            result["search_time_s"] = elapsed
            if n_routes > 0:
                # Take best route
                top = finder.routes[0]
                state_score = top.get("score", None)
                if state_score is None and "scores" in top:
                    state_score = top["scores"].get("state score")
                tree_dict = top["reaction_tree"].to_dict() if "reaction_tree" in top else None
                # Count reactions
                n_steps = 0
                if tree_dict:
                    def count_reactions(node):
                        if node is None: return 0
                        c = 1 if node.get("type") == "reaction" else 0
                        for ch in node.get("children", []):
                            c += count_reactions(ch)
                        return c
                    n_steps = count_reactions(tree_dict)
                result["top_route_state_score"] = state_score
                result["top_route_n_reactions"] = n_steps
                result["top_route_tree"] = tree_dict
                print(f"  -> {n_routes} routes; best score={state_score} steps={n_steps} ({elapsed:.0f}s)")
            else:
                print(f"  -> NO routes found within {args.max_iterations} iterations / {args.time_limit}s")
            sys.stdout.flush()
        except Exception as e:
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            print(f"  ERROR: {e}"); sys.stdout.flush()
        results.append(result)

    Path(args.out).write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[train] -> {args.out}")
    print("[train] === DONE ===")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
