"""Session state helpers for the Streamlit GUI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import MutableMapping, Sequence


@dataclass(frozen=True)
class CarbonSessionState:
    """Encapsulate the session state keys used by the carbon policy widgets."""

    enabled: bool
    price_enabled: bool
    enable_floor: bool
    enable_ccr: bool
    ccr1_enabled: bool
    ccr2_enabled: bool
    banking_enabled: bool
    bank0: float
    control_override: bool
    control_years: int
    ccr1_price: float
    ccr1_escalator: float
    ccr2_price: float
    ccr2_escalator: float
    coverage_regions: Sequence[str]
    price_value: float
    price_escalator: float
    cap_start: int
    cap_reduction_mode: str
    cap_reduction_percent: float
    cap_reduction_fixed: float
    floor_value_input: str
    floor_mode: str
    floor_escalator_input: str

    _SIGNATURE_KEY = "_carbon_defaults_signature"

    def apply_defaults(self, session: MutableMapping[str, object]) -> None:
        """Ensure session keys exist and refresh them when defaults change."""

        signature = self._signature()
        previous = session.get(self._SIGNATURE_KEY)
        overwrite = previous != signature
        self._assign(session, overwrite=bool(overwrite))
        session[self._SIGNATURE_KEY] = signature

    def override_for_lock(self, session: MutableMapping[str, object]) -> None:
        """Force session keys to the configured defaults when inputs are locked."""

        self._assign(session, overwrite=True)
        session[self._SIGNATURE_KEY] = self._signature()

    def _assign(self, session: MutableMapping[str, object], *, overwrite: bool) -> None:
        if overwrite:
            setter = session.__setitem__
        else:
            setter = lambda key, value: session.setdefault(key, value)

        setter("carbon_enable", bool(self.enabled))
        setter("carbon_price_enable", bool(self.price_enabled))
        setter("carbon_floor", bool(self.enable_floor))
        setter("carbon_ccr", bool(self.enable_ccr))
        setter("carbon_ccr1", bool(self.ccr1_enabled))
        setter("carbon_ccr2", bool(self.ccr2_enabled))
        setter("carbon_banking", bool(self.banking_enabled))
        setter("carbon_bank0", float(self.bank0))
        setter("carbon_control_toggle", bool(self.control_override))
        setter("carbon_control_years", int(self.control_years))
        setter("carbon_ccr1_price", float(self.ccr1_price))
        setter("carbon_ccr1_escalator", float(self.ccr1_escalator))
        setter("carbon_ccr2_price", float(self.ccr2_price))
        setter("carbon_ccr2_escalator", float(self.ccr2_escalator))
        setter("carbon_coverage_regions", list(self.coverage_regions))
        setter("carbon_price_value", float(self.price_value))
        setter("carbon_price_escalator", float(self.price_escalator))
        setter("carbon_cap_start", int(self.cap_start))
        setter("carbon_cap_reduction_mode", str(self.cap_reduction_mode))
        setter("carbon_cap_reduction_percent", float(self.cap_reduction_percent))
        setter("carbon_cap_reduction_fixed", float(self.cap_reduction_fixed))
        setter("carbon_floor_value_input", str(self.floor_value_input))
        setter("carbon_floor_mode", str(self.floor_mode))
        setter("carbon_floor_escalator_input", str(self.floor_escalator_input))

    def _signature(self) -> tuple[object, ...]:
        return (
            bool(self.enabled),
            bool(self.price_enabled),
            bool(self.enable_floor),
            bool(self.enable_ccr),
            bool(self.ccr1_enabled),
            bool(self.ccr2_enabled),
            bool(self.banking_enabled),
            float(self.bank0),
            bool(self.control_override),
            int(self.control_years),
            float(self.ccr1_price),
            float(self.ccr1_escalator),
            float(self.ccr2_price),
            float(self.ccr2_escalator),
            tuple(self.coverage_regions),
            float(self.price_value),
            float(self.price_escalator),
            int(self.cap_start),
            str(self.cap_reduction_mode),
            float(self.cap_reduction_percent),
            float(self.cap_reduction_fixed),
            str(self.floor_value_input),
            str(self.floor_mode),
            str(self.floor_escalator_input),
        )
