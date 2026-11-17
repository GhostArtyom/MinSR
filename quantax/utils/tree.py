from typing import Tuple
from jaxtyping import PyTree
import jax
import jax.tree_util as jtu
import jax.flatten_util as jfu
import jax.tree_util as jtu
from jax.lax import with_sharding_constraint
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import equinox as eqx


def tree_fully_flatten(tree: PyTree) -> jax.Array:
    array, unravel_fn = jfu.ravel_pytree(tree)
    return array


def filter_shard(tree: PyTree) -> PyTree:
    mesh = Mesh(np.array(jax.local_devices()), ("devices",))
    vals, tree_def = jtu.tree_flatten(tree)
    new_vals = []
    for v in vals:
        if eqx.is_array(v):
            spec = P("devices") if v.ndim else P()
            new_vals.append(with_sharding_constraint(v, NamedSharding(mesh, spec)))
        else:
            new_vals.append(v)
    return jtu.tree_unflatten(tree_def, new_vals)


def filter_replicate(tree: PyTree) -> PyTree:
    sharding = NamedSharding(Mesh(np.array(jax.local_devices()), ("devices",)), P())
    dynamic, static = eqx.partition(tree, eqx.is_array)
    dynamic = with_sharding_constraint(dynamic, sharding)
    return eqx.combine(dynamic, static)


def tree_split_cpl(tree: PyTree) -> Tuple[PyTree, PyTree]:
    get_real = lambda x: x.real if eqx.is_inexact_array(x) else x
    get_imag = lambda x: x.imag if eqx.is_inexact_array(x) else x
    tree_real = jtu.tree_map(get_real, tree)
    tree_imag = jtu.tree_map(get_imag, tree)
    return tree_real, tree_imag


def tree_combine_cpl(tree_real: PyTree, tree_imag: PyTree) -> PyTree:
    get_cpl = lambda x, y: x + 1j * y if eqx.is_inexact_array(x) else x
    return jtu.tree_map(get_cpl, tree_real, tree_imag)
