"""Block until a group of pueue tasks needs attention, then exit with a one-screen report.

usage: python qwatch.py LABEL_REGEX --eta MIN [--from ID] [--every SEC]
Watches tasks whose label matches LABEL_REGEX (and id >= --from). Exits on the FIRST of:
  failure  - any matching task finished with a non-Success result (prints its log tail), exit 1
  done     - every matching task finished Success, exit 0
  overdue  - --eta minutes passed (prints running/queued counts), exit 2
Run it as a background command so the agent is woken on exit instead of polling.
"""
import argparse
import json
import re
import subprocess
import sys
import time

ap = argparse.ArgumentParser(); ap.add_argument("label"); ap.add_argument("--eta", type=float, required=True)
ap.add_argument("--from", dest="first", type=int, default=0); ap.add_argument("--every", type=float, default=30)
a = ap.parse_args(); deadline, pat = time.time() + 60 * a.eta, re.compile(a.label)


def tasks():
    t = json.loads(subprocess.run(["pueue", "status", "--json"], capture_output=True, text=True).stdout)["tasks"]
    return {int(k): v for k, v in t.items() if int(k) >= a.first and pat.search(v.get("label") or "")}


while True:
    ts = tasks(); state = {k: next(iter(v["status"])) for k, v in ts.items()}
    bad = [k for k, v in ts.items() if state[k] == "Done" and v["status"]["Done"]["result"] != "Success"]
    if bad:
        print(f"FAILED {len(bad)}/{len(ts)}:", ", ".join(f"{k} {ts[k]['label']}" for k in bad))
        print(subprocess.run(["pueue", "log", str(bad[0]), "--lines", "15"], capture_output=True, text=True).stdout[-2500:])
        sys.exit(1)
    if ts and all(s == "Done" for s in state.values()):
        print(f"DONE {len(ts)} tasks"); sys.exit(0)
    if time.time() > deadline:
        c = {s: list(state.values()).count(s) for s in set(state.values())}
        print(f"OVERDUE after {a.eta:g} min: {c}"); sys.exit(2)
    time.sleep(a.every)
