# GPU job queue (2026-10-20)
- `runner.sh`: one job at a time from `pending/` (sorted by name: `<PP>_<name>.sh`, PP = priority).
- A job waits while any podman container runs (so it never overlaps another GPU user).
- `touch STOP` to stop the runner after the current job. Log: `runner.log`. Per-job logs go to done/ or failed/.
- Helpers: `/home/chronum/scratch/jax-rocm-wsl/run-apax.sh` (training), `<proj>/md.sh` (apax MD),
  `/home/chronum/scratch/petjax/run_gpu.sh` (pet-jax on the GPU).
