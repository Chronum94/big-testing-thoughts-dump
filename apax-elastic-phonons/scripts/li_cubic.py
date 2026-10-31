"""bcc Li with cubic symmetry kept (hydrostatic cell relax): the full-cell run let the teacher slide bcc -> fct (Bain path)."""
import json, sys
import elastic
elastic.CUBIC["Li"] = ("bcc", 3.44)
kind, path, out = sys.argv[1:4]
r = elastic.cij("Li", elastic.make_calc(kind, path))
r["model"] = path.rstrip("/").split("/")[-1]
open(out, "a").write(json.dumps(r) + "\n")
print(r["model"], round(r["a"], 4), {k: round(v, 2) for k, v in r["relaxed"].items() if k in ("C11", "C12", "C44", "B")}, "C'=", round((r["relaxed"]["C11"] - r["relaxed"]["C12"]) / 2, 2))
