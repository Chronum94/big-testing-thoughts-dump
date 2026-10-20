"""Work around a ROCm GPU sort bug (seen on gfx1201 under WSL): XLA's own sort kernel, used for n <= 16384 and for
every 64-bit-key argsort, returns wrong orders (64-bit keys: almost always; 32-bit: ~0.4% of calls, far more when
the GPU is shared). jax-md's cell list argsorts cell hashes, so a bad sort silently loses atoms from cells ->
wrong forces -> NaN after a few ps.
Patch: inside jax_md.partition, 1-D integer argsort runs on the host with NumPy's stable argsort. It only runs on
neighbour-list rebuilds, so the round trip (~5 ms) is small next to the MD steps."""
import jax
import jax.numpy as _jnp
import numpy as _np
from jax_md import partition as _partition


def host_argsort(a):
    idx = jax.dtypes.canonicalize_dtype(_jnp.int64)  # int32 when x64 is off
    out = jax.ShapeDtypeStruct(a.shape, idx)
    return jax.pure_callback(lambda x: _np.argsort(x, kind="stable").astype(idx), out, a)


class _JnpPatched:
    def __getattr__(self, name):
        return getattr(_jnp, name)

    @staticmethod
    def argsort(a, *args, **kwargs):
        if _jnp.issubdtype(a.dtype, _jnp.integer) and a.ndim == 1 and not args and not kwargs:
            return host_argsort(a)
        return _jnp.argsort(a, *args, **kwargs)


_partition.jnp = _JnpPatched()
