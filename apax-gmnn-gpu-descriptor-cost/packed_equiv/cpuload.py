import jax, jax.numpy as jnp, orbax.checkpoint as ocp
from pathlib import Path
import apax.train.checkpoints as ck
def load_params(model_version_path, best=True):
    p=(Path(model_version_path)/("best" if best else "")).resolve()
    step=max(int(s.name) for s in p.iterdir() if s.name.isdigit())
    d=p/str(step)/"default"
    c=ocp.StandardCheckpointer()
    meta=c.metadata(d)
    tree=getattr(meta,"item_metadata",meta)
    tree=getattr(tree,"tree",tree)
    shd=jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])
    tgt=jax.tree.map(lambda m: jax.ShapeDtypeStruct(m.shape,m.dtype,sharding=shd), tree)
    raw=c.restore(d,tgt)
    return jax.tree.map(jnp.asarray, raw["model"]["params"])
ck.load_params=load_params
