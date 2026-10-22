# crossed evaluation: every arm x every test set -> crossed_eval.md (all-atom force split + E MAE)
cd /home/chronum/scratch/ecemc
export PYTHONPATH=/home/chronum/apax-wt/avgnbr JAX_PLATFORMS=cpu XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
for arm in dup scan aff fresh ortho gen; do for t in md scan ortho gen; do
  f=logs_eval/${arm}_${t}.txt; [ -s $f ] && continue; mkdir -p logs_eval
  taskset -c 0-3 ~/miniconda3/envs/apaxenv/bin/python eval_inter_intra.py models/arm_$arm data/test_$t.extxyz 2>&1 | grep -v -i warn > $f
done; done
~/miniconda3/envs/apaxenv/bin/python - <<'PY'
import glob, re
arms, tests = ["dup", "scan", "aff", "fresh", "ortho", "gen"], ["md", "scan", "ortho", "gen"]
def get(a, t):
    s = open(f"logs_eval/{a}_{t}.txt").read(); m = re.search(r"^\| all \| \d+ \| ([\d.]+) \([\d.]+\) \| ([\d.]+) \([\d.]+\) \| ([\d.]+)", s, re.M)
    e = re.findall(r"^\| [\d.]+ \| \d+ \| [-+\d.]+ \| ([\d.]+) \|", s, re.M); md = re.search(r"MAE ([\d.]+) meV/atom", s)
    emae = float(md.group(1)) if md else (sum(map(float, e)) / len(e) if e else float("nan"))
    return [float(x) for x in m.groups()] + [emae]
out = ["| arm | " + " | ".join(f"{t}: trans / rot / intra / E" for t in tests) + " |", "|---|" + "---|" * len(tests)]
for a in arms:
    out.append(f"| {a} | " + " | ".join("{:.1f} / {:.1f} / {:.1f} / {:.2f}".format(*get(a, t)) for t in tests) + " |")
open("crossed_eval.md", "w").write("Force RMSE (meV/A) all atoms; E MAE (meV/atom; scan-type sets: mean over s)\n\n" + "\n".join(out) + "\n"); print("\n".join(out))
PY
