> **Note:** these tools were written with Claude (Anthropic's AI assistant) under close human oversight.

# Teacher → student tooling (PET-MAD via pet-jax → apax), single AMD GPU in podman on WSL2

These scripts are used by `../apax-kpba-bootstrap`, `../apax-superionic-water` and `../apax-lgps-bootstrap`. The paths are local to the machine they ran on.

| file | what |
|---|---|
| `runner.sh`, `gpuq_README.md` | one-GPU job queue: `pending/<PP>_<name>.sh`, `# MAXMIN=` cap, `STOP` file |
| `guard.sh` | `G <min> cmd`: a timeout that kills the job's containers. `Gw <min> <idle_min> <file> cmd` also kills them when `<file>` stops changing |
| `teacher_md.py` | teacher Langevin MD (test sets) |
| `student_ase_md.py` | one student Langevin run (ASE) for active-learning sampling, with a blow-up check |
| `sample_many.py` | many student runs in one process (one container start); resumes by skipping finished runs |
| `label_extxyz.py` | teacher relabelling, frame by frame; resumes from frames already written |
| `label_by_size.sh` | labels each atom count in its own container, with up to 3 retries |
| `al_select.py` | student-vs-teacher error tables per run and per element; top-error + random selection |
| `filter.py`, `split.py` | force/info filters; train/val/test split |
| `analyze_h5.py`, `label_h5.py` | apax MD trajectory analysis (MSD, D, closest pairs); teacher labels on h5 frames |

## Lessons

- **GPU hangs are random host-driver crashes,** not a problem in our code. A hung process sits in a device sync at ~300 % CPU and ignores SIGTERM. So:
  - every expensive step writes as it goes and resumes;
  - the guard kills containers, not processes;
  - an idle watchdog catches a hang long before the total timeout would.
- **Teacher speed:** PET-MAD-m single points are ~33–74 ms (50–200 atoms) at ambient density, and about 3.5× more per atom on dense water. Labelling measured ~1.6 s/frame, which is most likely neighbour-list rebuilds and recompiles at changing shapes. Sorting frames by size helps.
- **Compile cache:** it only helps when the program is identical (seeds, restarts, MD at a new T). The learning rate, loss weights and dataset size (with `scan_epoch`) are baked into the program as constants.
- **Scanned epoch on LGPS:** 2.27× faster per epoch, with a 5.2× longer first compile. Dispatch overhead ≈ 1.2× the step itself, so a 16–32-step scan chunk gets within 4–8 % of the full-epoch scan.
