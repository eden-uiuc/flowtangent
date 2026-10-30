from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from flowtangent.utils import field

from .nodes import EnergyStore, PACTNode, Splitter

# ----------------------------------------------------------------------------------------------------------------------
#  Energy Line
# ----------------------------------------------------------------------------------------------------------------------


class PACTLine(PACTNode):
    name: str | None = field("Line", static=True)
    _bookkeeping: dict = field(
        lambda: {
            "splitters": Splitter,
            "stores": EnergyStore,
        },
        static=True,
    )
