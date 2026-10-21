#!/usr/bin/env bash
# Single GPU job runner. Jobs: pending/<PP>_<name>.sh (PP = 2-digit priority, lower first; ties by name).
# Each job runs to completion (one GPU job at a time), then moves to done/ or failed/ with its log.
# A job may set "# MAXMIN=<minutes>" (default 240): past that, its containers are killed and it fails.
# Starts only once the given wait-file condition holds (first arg: a file that must contain "queue5 done" or be absent).
cd /home/chronum/scratch/gpuq
export XDG_RUNTIME_DIR=/run/user/1000
log() { echo "$(date '+%F %T') $*" >> runner.log; }
log "runner start pid $$"
while true; do
  [ -f STOP ] && { log "STOP file, exiting"; exit 0; }
  if [ -n "$(podman ps -q)" ]; then sleep 15; continue; fi   # something else holds the GPU
  job=$(ls pending/*.sh 2>/dev/null | sort | head -n 1)
  if [ -z "$job" ]; then sleep 15; continue; fi
  name=$(basename "$job" .sh); mv "$job" running/
  max=$(grep -oP '^# MAXMIN=\K[0-9]+' "running/$name.sh" || echo 240)
  log "start $name (max ${max} min)"; t0=$(date +%s)
  bash "running/$name.sh" > "running/$name.log" 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 20
    if [ $(( $(date +%s) - t0 )) -gt $(( max * 60 )) ]; then
      log "timeout $name, killing"; kill $pid; for c in $(podman ps -q); do podman kill "$c" >/dev/null; done
    fi
  done
  wait $pid; rc=$?
  dest=done; [ $rc -ne 0 ] && dest=failed
  mv "running/$name.sh" "running/$name.log" $dest/
  log "end $name rc $rc $(( ($(date +%s) - t0) / 60 )) min -> $dest"
done
