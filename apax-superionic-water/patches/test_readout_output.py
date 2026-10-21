import jax
import jax.numpy as jnp
import numpy as np
from flax import traverse_util

from apax.layers.readout import OUTPUT_ACTIVATIONS, AtomisticReadout


def test_zero_init_output_starts_at_zero_and_trains():
    x = jax.random.normal(jax.random.PRNGKey(0), (5, 8))
    readout = AtomisticReadout(units=[16, 16], zero_init_output=True, output_activation="asinh")
    params = readout.init(jax.random.PRNGKey(1), x[0])
    np.testing.assert_array_equal(jax.vmap(lambda xi: readout.apply(params, xi))(x), 0.0)

    grads = jax.grad(lambda p: jnp.sum(jax.vmap(lambda xi: readout.apply(p, xi))(x) - 1.0) ** 2)(params)
    flat = traverse_util.flatten_dict(grads)
    (w_out,) = [g for path, g in flat.items() if "dense_2" in path]
    assert np.abs(w_out).max() > 0  # the floor-free output passes gradient at 0


def test_asinh_output_has_unit_slope_at_zero():
    assert jax.grad(OUTPUT_ACTIVATIONS["asinh"])(0.0) == 1.0
