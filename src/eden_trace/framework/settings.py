# Trace/Framework/Settings.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, Trace Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from typing import Optional, Literal

import os
import logging

from pathlib import Path

# package imports
import equinox as eqx
import jax

# Trace imports
from eden_trace.utils import init_field
from eden_trace.framework.processes import GradientMap

# ----------------------------------------------------------------------------------------------------------------------
#  Settings
# ----------------------------------------------------------------------------------------------------------------------


# Analysis Settings ----------------------------------------------------------------------------------------------------


# Mass Analysis ----------------------------------------------------------------


class ReductionFactors(eqx.Module):
    main_wing: float = 0.0
    fuselage: float = 0.0
    empennage: float = 0.0
    systems: float = 0.0


class SizingFractions(eqx.Module):
    rudder_sizing: float = 0.25


class MassAnalysisSettings(eqx.Module):
    reduction_factors: ReductionFactors = init_field(ReductionFactors)
    sizing_fractions: SizingFractions = init_field(SizingFractions)

# Energy Analysis --------------------------------------------------------------

class EnergyAnalysisSettings(eqx.Module):
    report_units: Literal["SI", "Imperial"] = init_field("SI", static=True)

    build_network: bool = init_field(True)
    clear_nodes: bool = init_field(True)


class AnalysisSettings[E_Type: EnergyAnalysisSettings](eqx.Module):
    aerodynamics: Optional[eqx.Module] = None
    energy: E_Type = init_field(EnergyAnalysisSettings)
    mass: MassAnalysisSettings = init_field(MassAnalysisSettings)

    gradient_map: Optional[GradientMap] = init_field(None, static=True)


#  Numerical Settings --------------------------------------------------------------------------------------------------

class NumericalSettings(eqx.Module):
    
    relative_tolerance: float = init_field(1e-5, static=True)
    absolute_tolerance: float = init_field(1e-5, static=True)
    
    max_evaluations: int = init_field(100, static=True)
    step_size: float | None = init_field(None, static=True)

    batch_size: int = init_field(1, static=True)
    batch_mode: Literal['zip', 'mesh'] = init_field('zip', static=True)

    number_of_control_points: int = init_field(1, static=True)
    maximum_graph_complexity: int = init_field(2e5, static=True)

    sum_residuals: bool = init_field(False, static=True)

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


class LoggingSettings(eqx.Module):

    handle: Optional[str] = init_field(None, static=True)
    logfile: Optional[str | Path] = init_field(None, static=True)

    format_string: str = init_field("[%(asctime)s] - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", static=True)
    stream_ouput: bool = init_field(False, static=True)
    jax_logging: bool = init_field(False, static=True)
    jax_compile_whitelist: Optional[tuple[str]] = init_field(None, static=True)

    def setup_logger(self, handle: Optional[str] = None) -> None:
        if self.logfile is None and not self.stream_ouput:
            return
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

            if self.logfile is not None:
                fh = logging.FileHandler(self.logfile)
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



class Settings(eqx.Module):
    tag: str = init_field("Settings", static=True)

    report_units: Literal["SI", "Imperial"] = init_field("SI", static=True)

    analysis: AnalysisSettings = init_field(AnalysisSettings)
    numerical: NumericalSettings = init_field(NumericalSettings)

    logging: LoggingSettings = init_field(LoggingSettings)

    DEBUG_MODE: bool = init_field(False, static=True)
    verbose: bool = init_field(False, static=True)
    JAX_device_index: int = init_field(0, static=True)

    _DEV_MODE: bool = init_field(False, static=True)

    def __post_init__(self):
        if self._DEV_MODE:
            os.environ["XLA_FLAGS"] = "--xla_backend_optimization_level=0"
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        if self.DEBUG_MODE:
            jax.config.update("jax_disable_jit", True)
            jax.config.update("jax_debug_nans", True)
            object.__setattr__(self, "verbose", True)
        else:
            # Manually reset flags
            jax.config.update("jax_disable_jit", False)
            jax.config.update("jax_debug_nans", False)
