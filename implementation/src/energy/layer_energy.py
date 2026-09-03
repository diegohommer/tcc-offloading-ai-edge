"""Loader for config/layer_energy.yaml -- the per-layer energy tables from
the TCC master document, section 6.

This module only parses and looks up values; it does not judge whether a
given lookup is a sound comparison (e.g. mixing a batch=1 number with a
batch=64 number). That judgment belongs to the caller and must be stated
explicitly in any report that uses these values -- see src/energy/cost.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_LAYER_ENERGY_PATH = Path(__file__).resolve().parents[2] / "config" / "layer_energy.yaml"


@dataclass
class ModelDecodePoint:
    layer: str
    model_key: str
    precision: str
    decode_J_per_token: float
    source: str


class LayerEnergyTable:
    def __init__(self, path: Path | str = DEFAULT_LAYER_ENERGY_PATH):
        self.path = Path(path)
        with open(self.path) as f:
            self._data: dict[str, Any] = yaml.safe_load(f)

    @property
    def layers(self) -> dict[str, Any]:
        return self._data["layers"]

    @property
    def link(self) -> dict[str, Any]:
        return self._data["link"]

    @property
    def environment(self) -> dict[str, Any]:
        return self._data["environment"]

    def decode_point(self, layer: str, model_key: str, precision: str) -> ModelDecodePoint:
        """Look up a measured decode J/token point for user/onu layers, where
        the table is keyed by explicit (model, precision) pairs."""
        layer_data = self.layers[layer]
        model_data = layer_data["models"][model_key]
        value = model_data["decode_J_per_token"][precision]
        return ModelDecodePoint(
            layer=layer,
            model_key=model_key,
            precision=precision,
            decode_J_per_token=float(value),
            source=layer_data["source"],
        )

    def fog_primary_J_per_token(self) -> float:
        return float(self.layers["fog"]["primary_point"]["decode_J_per_token"])

    def fog_cross_check_upper_bound_J_per_token(self) -> float:
        return float(self.layers["fog"]["cross_check"]["decode_J_per_token_upper_bound"])

    def cloud_isolated_query_J_per_token(self) -> tuple[float, float]:
        lo, hi = self.layers["cloud"]["isolated_query_regime"]["decode_J_per_token_range"]
        return float(lo), float(hi)

    def cloud_production_J_per_token(self) -> float:
        return float(self.layers["cloud"]["production_regime"]["ml_energy_v3"]["derived_J_per_token"])

    def link_energy_per_hop_J(self, rate: str = "twdm_pon_25_Gbps_per_lambda") -> float:
        """Energia MARGINAL de transporte por salto, em joules.

        Este e o custo de mandar uma consulta a mais, e e o numero certo para a
        decisao de escalonar: numa PON a ONU esta ligada de qualquer forma, entao
        o custo incremental e apenas tempo de transmissao vezes potencia.

        NAO confundir com o custo amortizado sempre-ligado (potencia da ONU
        dividida por consultas por segundo), que e da ordem de joules e depende
        inteiramente da carga da residencia, nao da cascata. Ver
        link_energy_amortised_J() e o bloco `link:` do layer_energy.yaml.

        Ate 2026-09-03 este metodo devolvia uma constante de 0.1 J que, apurada
        depois, era um valor amortizado a 39.8 consultas por segundo -- premissa
        de carga que nunca esteve documentada.
        """
        return float(self.link["marginal"]["J_per_hop"][rate])

    def link_energy_amortised_J(self, queries_per_second: float) -> float:
        """Parcela da potencia sempre-ligada da ONU atribuida a uma consulta.

        Para contabilidade total do sistema, nao para a decisao de escalonar.
        Exige declarar a taxa de consultas assumida: o valor varia cinco ordens
        de magnitude entre uma consulta por hora e quarenta por segundo.
        """
        return float(self.link["onu_power_active_W"]) / queries_per_second

    def pue(self, kind: str = "average") -> float:
        return float(self.environment["pue"][kind])
