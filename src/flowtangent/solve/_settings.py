from typing import Literal, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .. import Module, TreePath, field, static_field, update
from ..utils import get_all_parents, get_all_targets
from ._mass import MassAnalysisSettings

# Energy Analysis --------------------------------------------------------------


class EnergyAnalysisSettings(Module):
    report_units: Literal["SI", "Imperial"] = static_field("SI")

    build_network: bool = field(True)
    clear_nodes: bool = field(True)


class AnalysisSettings[E_Type: EnergyAnalysisSettings](Module):
    aerodynamics: Optional[Module] = None
    energy: E_Type = field(EnergyAnalysisSettings)
    mass: MassAnalysisSettings = field(MassAnalysisSettings)


#  Numerical Settings --------------------------------------------------------------------------------------------------


class JacobianMap(Module):
    inputs: tuple[TreePath, ...] = static_field(())
    outputs: tuple[TreePath, ...] = static_field(())

    state_inputs: tuple[TreePath, ...] = static_field(())
    state_outputs: tuple[TreePath, ...] = static_field(())

    system_inputs: tuple[TreePath, ...] = static_field(())
    system_outputs: tuple[TreePath, ...] = static_field(())

    _n_st: int = static_field(0)
    _n_sys: int = static_field(0)

    def __init__(
        self,
        inputs: tuple[TreePath | str | tuple, ...] = (),
        outputs: tuple[TreePath | str | tuple, ...] = (),
        state_inputs: Optional[tuple] = None,
        state_outputs: Optional[tuple] = None,
        system_inputs: Optional[tuple] = None,
        system_outputs: Optional[tuple] = None,
    ):
        self.inputs = tuple(TreePath(i) for i in inputs)
        self.outputs = tuple(TreePath(o) for o in outputs)

        _filter_in = lambda s: tuple(p for p in self.inputs if p.path[0].lower() == s) #noqa: E731
        _filter_out = lambda s: tuple(p for p in self.outputs if p.path[0].lower() == s) #noqa: E731

        self.state_inputs   = _filter_in("state") if state_inputs is None else state_inputs
        self.system_inputs  = _filter_in("system") if system_inputs is None else system_inputs
        self.state_outputs  = _filter_out("state") if state_outputs is None else state_outputs
        self.system_outputs = _filter_out("system") if system_outputs is None else system_outputs

        self._n_st = len(self.state_inputs)
        self._n_sys = len(self.system_inputs)

    def flatten_inputs(self, base_state, base_system):
        # Helper to find leading dimensions (e.g., B and T) safely
        def _get_leading(tree):
            arr = next((l for l in jax.tree_util.tree_leaves(tree) if isinstance(l, (jax.Array, np.ndarray))), None)
            return arr.shape[:-1] if arr is not None else ()

        flat_st = []
        if self._n_st > 0:
            st_in = get_all_targets(base_state, self.state_inputs)
            # FIX 2: Preserve leading dims, flatten trailing
            flat_st = [x.reshape(*x.shape[:-1], -1) for x in st_in]
        flat_st_array = jnp.concatenate(flat_st, axis=-1) if flat_st else jnp.empty((*_get_leading(base_state), 0))

        flat_sys = []
        if self._n_sys > 0:
            sys_in = get_all_targets(base_system, self.system_inputs)
            flat_sys = [x.reshape(*x.shape[:-1], -1) for x in sys_in]
        flat_sys_array = jnp.concatenate(flat_sys, axis=-1) if flat_sys else jnp.empty((*_get_leading(base_system), 0))

        return flat_st_array, flat_sys_array

    def update_inputs(self, flat_st, flat_sys, base_state, base_system):
        st, sys = base_state, base_system

        if self._n_st > 0:
            st_in = get_all_targets(st, self.state_inputs)
            shapes = [x.shape for x in st_in]
            sizes = [x.shape[-1] for x in st_in] # Number of features per array

            # Split along the feature axis
            splits = jnp.split(flat_st, np.cumsum(sizes)[:-1], axis=-1)
            new_slices = [s.reshape(shp) for s, shp in zip(splits, shapes)]

            parents = get_all_parents(st, self.state_inputs)
            updated = [
                p.at[pth.path_slice].set(n) if pth.path_slice != slice(None) else n
                for p, n, pth in zip(parents, new_slices, self.state_inputs)
            ]
            st = update(st, lambda t: get_all_parents(t, self.state_inputs), tuple(updated))

        if self._n_sys > 0:
            sys_in = get_all_targets(sys, self.system_inputs)
            shapes = [x.shape for x in sys_in]
            sizes = [x.shape[-1] for x in sys_in]

            splits = jnp.split(flat_sys, np.cumsum(sizes)[:-1], axis=-1)
            new_slices = [s.reshape(shp) for s, shp in zip(splits, shapes)]

            parents = get_all_parents(sys, self.system_inputs)
            updated = [
                p.at[pth.path_slice].set(n) if pth.path_slice != slice(None) else n
                for p, n, pth in zip(parents, new_slices, self.system_inputs)
            ]
            sys = update(sys, lambda t: get_all_parents(t, self.system_inputs), tuple(updated))

        return st, sys

    def flatten_outputs(self, f_st, f_sys, f_setts):
        outputs = []
        if self.state_outputs:
            outputs.extend(get_all_targets(f_st, self.state_outputs))
        if self.system_outputs:
            outputs.extend(get_all_targets(f_sys, self.system_outputs))

        if not outputs:
            return jnp.empty((0,))

        # Concatenate along the feature dimension, preserving B and T dynamically
        return jnp.concatenate([out.reshape(*out.shape[:-1], -1) for out in outputs], axis=-1)

class JacobianSettings(Module):
    calculate: bool =   static_field(False)
    couple_time: bool = static_field(True)
    mapping: Optional[JacobianMap] = static_field(None)


class NumericalSettings(Module):
    relative_tolerance: float = static_field(1e-5)
    absolute_tolerance: float = static_field(1e-5)

    max_evaluations: int = static_field(100)
    step_size: float | None = static_field(None)

    batch_size: int = static_field(1)
    batch_mode: Literal["zip", "mesh"] = static_field("zip")

    partition_inputs: bool = static_field(True)
    sum_residuals: bool = static_field(False)

    number_of_control_points: int = static_field(1)
    maximum_graph_complexity: int = static_field(int(1e6))

    jacobian: JacobianSettings = field(JacobianSettings)
