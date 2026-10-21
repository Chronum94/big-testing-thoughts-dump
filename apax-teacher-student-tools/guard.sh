# Source in GPU jobs: G <minutes> cmd... runs cmd under a timeout; on timeout it kills the job's containers
# (a GPU-hung process ignores SIGTERM) so the job moves on to its next step.
G() { local m=$1; shift; timeout -k 60 "${m}m" "$@"; local rc=$?
  if [ $rc -ge 124 ]; then echo "GUARD timeout ${m}m rc $rc: $*" >&2; podman ps -q | xargs -r podman kill >/dev/null; fi
  return $rc; }
# Gw <minutes> <idle_minutes> <file> cmd...: like G, but also kills the job's containers when <file> has not been modified
# for <idle_minutes> (a hang from a host GPU-driver crash makes no progress long before the total timeout).
Gw() { local m=$1 idle=$2 f=$3; shift 3; timeout -k 60 "${m}m" "$@" & local pid=$! t=$(date +%s)
  while kill -0 $pid 2>/dev/null; do sleep 20
    local last=$(stat -c %Y "$f" 2>/dev/null || echo $t); [ $last -lt $t ] && last=$t
    if [ $(( $(date +%s) - last )) -gt $(( idle * 60 )) ]; then echo "GUARD idle ${idle}m on $f: $*" >&2; podman ps -q | xargs -r podman kill >/dev/null; fi
  done
  wait $pid; local rc=$?
  [ $rc -ge 124 ] && { echo "GUARD stop rc $rc: $*" >&2; podman ps -q | xargs -r podman kill >/dev/null; }
  return $rc; }
