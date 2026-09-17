# flowtangent/Framework/Settings.py
# (c) Copyright 2024 Aerospace Research Community LLC
#
# Created: Jul 2024, Flowtangent Team

# ----------------------------------------------------------------------------------------------------------------------
#  IMPORT
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from typing import Literal, Optional

from ..solve import AnalysisSettings, NumericalSettings

# Flowtangent imports
from ..utils import LoggingSettings, Module, field, static_field

# ----------------------------------------------------------------------------------------------------------------------
#  Settings
# ----------------------------------------------------------------------------------------------------------------------


#  Full Settings -------------------------------------------------------------------------------------------------------


class Settings(Module):
    name: Optional[str] = static_field("Settings")

    report_units: Literal["SI", "Imperial"] = static_field("SI")

    analysis: AnalysisSettings = field(AnalysisSettings)
    numerical: NumericalSettings = field(NumericalSettings)

    logging: LoggingSettings = field(LoggingSettings)
    verbose: bool = static_field(False)

    _DEV_MODE: bool = static_field(False)
    DEBUG_MODE: bool = static_field(False)
    JAX_device_index: int = static_field(0)


