# Trace/Framework/Settings.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, Trace Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations
from typing import Optional, Literal, TYPE_CHECKING
if TYPE_CHECKING:
    from . import State, System, Settings

from datetime import datetime
import logging

from pathlib import Path

# package imports
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np # For calculating Jacobian shape on JAX array metadata

# Trace imports
from eden_trace.utils import init_field, DataPath, get_all_targets, get_all_parents

# ----------------------------------------------------------------------------------------------------------------------
#  Settings
# ----------------------------------------------------------------------------------------------------------------------

class StaticModuleMeta(type(eqx.Module)):
    def __new__(mcs, name, bases, namespace):
        annotations = namespace.get('__annotations__', {})
        for key in annotations:
            # Skip internal python/equinox variables
            if key.startswith("__"): continue
            
            # If they provided a default (e.g. `val: int = 5`), wrap it
            if key in namespace:
                val = namespace[key]
                # Don't double-wrap if they're already fields
                if not getattr(val, 'metadata', None): 
                    namespace[key] = init_field(val, static=True)
            # If there's no default (e.g. `val: int`), inject an empty static field
            else:
                namespace[key] = eqx.field(static=True)
                
        return super().__new__(mcs, name, bases, namespace)

class StaticModule(eqx.Module, metaclass=StaticModuleMeta):
    pass

# Analysis Settings ----------------------------------------------------------------------------------------------------


# Mass Analysis ----------------------------------------------------------------


class ReductionFactors(StaticModule):
    main_wing:  float = 0.0
    fuselage:   float = 0.0
    empennage:  float = 0.0
    systems:    float = 0.0


class SizingFractions(StaticModule):
    rudder_sizing: float = init_field(0.25, static=True)


class MassAnalysisSettings(StaticModule):
    reduction_factors: ReductionFactors = init_field(ReductionFactors)
    sizing_fractions: SizingFractions = init_field(SizingFractions)

# Energy Analysis --------------------------------------------------------------

class EnergyAnalysisSettings(StaticModule):
    report_units: Literal["SI", "Imperial"] = init_field("SI", static=True)

    build_network: bool = init_field(True)
    clear_nodes: bool = init_field(True)


class AnalysisSettings[E_Type: EnergyAnalysisSettings](StaticModule):
    aerodynamics: Optional[eqx.Module] = None
    energy: E_Type = init_field(EnergyAnalysisSettings)
    mass: MassAnalysisSettings = init_field(MassAnalysisSettings)


#  Numerical Settings --------------------------------------------------------------------------------------------------

class JacobianMap(StaticModule):
    inputs: tuple[DataPath, ...] = eqx.field(static=True)
    outputs: tuple[DataPath, ...] = eqx.field(static=True)

    state_inputs: tuple[DataPath, ...] = eqx.field(static=True)
    state_outputs: tuple[DataPath, ...] = eqx.field(static=True)

    system_inputs: tuple[DataPath, ...] = eqx.field(static=True)
    system_outputs: tuple[DataPath, ...] = eqx.field(static=True)

    _n_st: int = eqx.field(static=True)
    _n_sys: int = eqx.field(static=True)

    def __init__(
        self,
        inputs: tuple[DataPath | str | tuple, ...] = (),
        outputs: tuple[DataPath | str | tuple, ...] = (),
        state_inputs: Optional[tuple] = None,
        state_outputs: Optional[tuple] = None,
        system_inputs: Optional[tuple] = None,
        system_outputs: Optional[tuple] = None,
    ):
        self.inputs = tuple(DataPath(i) for i in inputs)
        self.outputs = tuple(DataPath(o) for o in outputs)

        self.state_inputs = tuple(p for p in self.inputs if p.path[0].lower()=="state") if state_inputs is None else state_inputs
        self.system_inputs = tuple(p for p in self.inputs if p.path[0].lower()=="system") if system_inputs is None else system_inputs
        self.state_outputs = tuple(p for p in self.outputs if p.path[0].lower()=="state") if state_outputs is None else state_outputs
        self.system_outputs = tuple(p for p in self.outputs if p.path[0].lower()=="system") if system_outputs is None else system_outputs

        self._n_st = len(self.state_inputs)
        self._n_sys = len(self.system_inputs)

    def flatten_inputs(self, base_state, base_system):
        # Dynamically detect B from the base state (if any arrays are 3D)
        arr = next((l for l in jax.tree_util.tree_leaves(base_state) if isinstance(l, (jax.Array, np.ndarray))), None)
        has_B = (arr is not None and arr.ndim == 3)
        B = arr.shape[0] if has_B else None

        flat_st = []
        if self._n_st > 0:
            st_in = get_all_targets(base_state, self.state_inputs)
            flat_st = [x.reshape(B, -1) if has_B else x.reshape(-1) for x in st_in]
        flat_st_array = jnp.concatenate(flat_st, axis=-1) if flat_st else jnp.empty((B, 0) if has_B else (0,))

        flat_sys = []
        if self._n_sys > 0:
            sys_in = get_all_targets(base_system, self.system_inputs)
            flat_sys = [x.reshape(-1) for x in sys_in]
        flat_sys_array = jnp.concatenate(flat_sys, axis=-1) if flat_sys else jnp.empty((0,))

        return flat_st_array, flat_sys_array

    def update_inputs(self, flat_st, flat_sys, base_state, base_system):
        has_B = (flat_st.ndim == 2)
        B = flat_st.shape[0] if has_B else None

        st, sys = base_state, base_system

        # Update State
        if self._n_st > 0:
            st_in = get_all_targets(st, self.state_inputs)
            shapes = [x.shape[1:] if has_B else x.shape for x in st_in]
            sizes = [int(np.prod(s)) if s else 1 for s in shapes]
            
            splits = jnp.split(flat_st, np.cumsum(sizes)[:-1], axis=-1)
            new_slices = [s.reshape((B,) + shp) if has_B else s.reshape(shp) for s, shp in zip(splits, shapes)]
            
            parents = get_all_parents(st, self.state_inputs)
            updated = [p.at[pth.path_slice].set(n) if pth.path_slice != slice(None) else n for p, n, pth in zip(parents, new_slices, self.state_inputs)]
            st = eqx.tree_at(lambda t: get_all_parents(t, self.state_inputs), st, tuple(updated))

        # Update System
        if self._n_sys > 0:
            sys_in = get_all_targets(sys, self.system_inputs)
            shapes = [x.shape for x in sys_in]
            sizes = [int(np.prod(s)) if s else 1 for s in shapes]
            
            splits = jnp.split(flat_sys, np.cumsum(sizes)[:-1], axis=-1)
            new_slices = [s.reshape(shp) for s, shp in zip(splits, shapes)]
            
            parents = get_all_parents(sys, self.system_inputs)
            updated = [p.at[pth.path_slice].set(n) if pth.path_slice != slice(None) else n for p, n, pth in zip(parents, new_slices, self.system_inputs)]
            sys = eqx.tree_at(lambda t: get_all_parents(t, self.system_inputs), sys, tuple(updated))

        return st, sys

    def flatten_outputs(self, f_st, f_sys, f_setts):
        outputs = []
        if self.state_outputs:
            outputs.extend(get_all_targets(f_st, self.state_outputs))
        if self.system_outputs:
            outputs.extend(get_all_targets(f_sys, self.system_outputs))
            
        has_B = (outputs[0].ndim == 3)
        B = outputs[0].shape[0] if has_B else None
        
        if has_B:
            return jnp.concatenate([out.reshape(B, -1) for out in outputs], axis=-1)
        else:
            return jnp.concatenate([out.reshape(-1) for out in outputs], axis=-1)

class JacobianSettings(StaticModule):

    calculate: bool = init_field(False, static=True)
    couple_time: bool = init_field(True, static=True)
    mapping: Optional[JacobianMap] = init_field(None, static=True)

class NumericalSettings(StaticModule):
    
    relative_tolerance: float = init_field(1e-5, static=True)
    absolute_tolerance: float = init_field(1e-5, static=True)
    
    max_evaluations: int = init_field(100, static=True)
    step_size: float | None = init_field(None, static=True)

    batch_size: int = init_field(1, static=True)
    batch_mode: Literal['zip', 'mesh'] = init_field('zip', static=True)

    number_of_control_points: int = init_field(1, static=True)
    maximum_graph_complexity: int = init_field(1e6, static=True)

    sum_residuals: bool = init_field(False, static=True)

    jacobian: JacobianSettings = init_field(JacobianSettings, static=True)

#  Numerical Settings --------------------------------------------------------------------------------------------------

class JAXCompileFilter(logging.Filter):

    def __init__(self, name: str = "", whitelist: Optional[tuple[str]] = None) -> None:
        super().__init__(name)
        self.whitelist = whitelist
    
    def filter(self, record):
        msg = record.getMessage()

        # 1. Identify if this is a compilation/tracing log
        is_compile_log = any(
            keyword in msg
            for keyword in ["Compiling", "tracing + transforming", "Finished jaxpr to MLIR", "Finished XLA compilation"]
        )

        # If it is a compile log, apply whitelist & formatting
        if is_compile_log and self.whitelist is not None:
            # Block it if it's not the main solve
            if not any([f"jit({w})" in msg for w in self.whitelist]):
                return False

            # If it is the main solve, truncate the massive PyTree dump
            if "with global shapes and types" in msg:
                parts = msg.split("with global shapes and types")
                prefix = parts[0] + "with global shapes and types"
                suffix = parts[1][:30] if len(parts) > 1 else ""

                record.msg = f"{prefix} {suffix} ... [PyTree Truncated]"
                record.args = ()

            return True

        # If it's NOT a compile log (e.g., GPU memory warning), let it through untouched
        return True


class LoggingSettings(StaticModule):

    handle: Optional[str] = init_field(None, static=True)
    log_dir: Optional[str | Path] = init_field(None, static=True)

    format_string: str = init_field("[%(asctime)s] - %(levelname)s - %(message)s", static=True)
    date_format: str = init_field("%Y-%m-%d %H:%M:%S", static=True)
    stream_ouput: bool = init_field(False, static=True)
    jax_logging: bool = init_field(False, static=True)
    jax_compile_whitelist: Optional[tuple[str]] = init_field(None, static=True)

    def setup_logger(self, handle: Optional[str] = None) -> None:
        if self.log_dir is None and not self.stream_ouput:
            return
        else:
            if handle is None and self.handle is None:
                log_handle = "flowtangent"
            else:
                log_handle = handle if handle is not None else self.handle
            logger = logging.getLogger(log_handle)
            formatter = logging.Formatter(self.format_string)
            handlers = []

            if self.stream_ouput:
                sh = logging.StreamHandler()
                sh.setLevel(logging.INFO)
                sh.setFormatter(formatter)
                handlers.append(sh)

            if self.log_dir is not None:
                log_dir = Path(self.log_dir)
                log_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime(self.date_format).replace(' ', '_').replace(':','-')
                logfile = log_dir / f"main_{timestamp}.log"
                fh = logging.FileHandler(logfile)
                fh.setLevel(logging.INFO)
                fh.setFormatter(formatter)
                handlers.append(fh)

            for h in handlers:
                logger.addHandler(h)

            if self.jax_logging:
                jl = logging.getLogger("jax")
                jl.propagate = False
                jl.handlers.clear()
                j_filter = JAXCompileFilter("jax_compile_filter", self.jax_compile_whitelist)

                for h in handlers:
                    h.addFilter(j_filter)
                    jl.addHandler(h)

                if getattr(jax.config, "jax_log_compiles", False):
                    jl.setLevel(logging.INFO)
                else:
                    jl.setLevel(logging.WARNING)

            return    

#  Full Settings -------------------------------------------------------------------------------------------------------

class Settings(StaticModule):
    tag: str = init_field("Settings", static=True)

    report_units: Literal["SI", "Imperial"] = init_field("SI", static=True)

    analysis: AnalysisSettings = init_field(AnalysisSettings)
    numerical: NumericalSettings = init_field(NumericalSettings)

    logging: LoggingSettings = init_field(LoggingSettings)

    DEBUG_MODE: bool = init_field(False, static=True)
    verbose: bool = init_field(False, static=True)
    JAX_device_index: int = init_field(0, static=True)

    _DEV_MODE: bool = init_field(False, static=True)
