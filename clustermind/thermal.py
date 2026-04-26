"""Thermal & cooling dynamics.

Pure functions that mutate node temperatures, cooling stress, and energy
consumption in place. The simulator drives them once per step.

PRD §14.5–14.7 specify the formulas. We keep them simple and clipped; the
goal is *interesting tradeoffs*, not high-fidelity HVAC.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from clustermind.models import (
    CoolingZone,
    GPUNode,
    INTENSITY_MULTIPLIER,
    IntensityLevel,
    NodeStatus,
    ZoneStatus,
)


BASE_COOLING_COST = 0.012
ENV_TEMP_FLOOR = 30.0
ENV_TEMP_CEIL = 110.0


# ---------------------------------------------------------------------------
# Per-zone effective efficiency
# ---------------------------------------------------------------------------

def effective_efficiency(zone: CoolingZone) -> float:
    return max(0.05, zone.cooling_efficiency * (1.0 - 0.3 * zone.cooling_stress))


def cooling_temperature_drop(zone: CoolingZone) -> float:
    intensity = INTENSITY_MULTIPLIER[zone.intensity]
    return 8.0 * zone.cooling_power * effective_efficiency(zone) * intensity


def cooling_energy_cost(zone: CoolingZone) -> float:
    intensity = INTENSITY_MULTIPLIER[zone.intensity]
    stress_multiplier = 1.0 + 0.4 * zone.cooling_stress
    return BASE_COOLING_COST * intensity * zone.energy_cost_multiplier * stress_multiplier


def apply_cooling_action(zone: CoolingZone, intensity: IntensityLevel) -> None:
    zone.intensity = intensity
    if intensity == IntensityLevel.HIGH:
        zone.cooling_stress = min(1.0, zone.cooling_stress + 0.05)
        zone.high_cooling_steps += 1
    elif intensity == IntensityLevel.MEDIUM:
        zone.high_cooling_steps = max(0, zone.high_cooling_steps - 1)
    else:  # LOW
        zone.high_cooling_steps = max(0, zone.high_cooling_steps - 1)
        zone.cooling_stress = max(0.0, zone.cooling_stress - 0.02)


def passive_cooling_recovery(zone: CoolingZone) -> None:
    """When intensity is medium/low for a while, stress slowly recovers."""

    if zone.intensity != IntensityLevel.HIGH:
        zone.cooling_stress = max(0.0, zone.cooling_stress - 0.01)


# ---------------------------------------------------------------------------
# Temperature update for one tick
# ---------------------------------------------------------------------------

def _neighbor_heat_pressure(node: GPUNode, by_id: Dict[str, GPUNode]) -> float:
    if not node.neighbors:
        return 0.0
    temps = [by_id[nid].temperature for nid in node.neighbors if nid in by_id]
    if not temps:
        return 0.0
    return sum(temps) / len(temps) - node.temperature


def update_node_temperature(
    node: GPUNode,
    zone: CoolingZone,
    by_id: Dict[str, GPUNode],
) -> None:
    """One-step temperature delta per PRD §14.7."""

    if node.status in (NodeStatus.FAILED, NodeStatus.SHUTDOWN):
        # Failed/shutdown nodes drift toward ambient.
        node.temperature += (40.0 - node.temperature) * 0.15
        node.temperature = _clamp_temp(node.temperature)
        return

    util_gain = 9.5 * max(0.0, node.utilization)
    neighbor_pressure = _neighbor_heat_pressure(node, by_id)
    neighbor_gain = 2.5 * max(0.0, neighbor_pressure / 30.0)
    cooling_drop = cooling_temperature_drop(zone)
    throttle_bonus = 4.0 if node.throttled else 0.0
    maintenance_bonus = 8.0 if node.status == NodeStatus.MAINTENANCE else 0.0

    new_temp = node.temperature + util_gain + neighbor_gain - cooling_drop - throttle_bonus - maintenance_bonus
    node.temperature = _clamp_temp(new_temp)

    # Update visible status thresholds (without changing failed/maintenance/shutdown).
    if node.status not in (NodeStatus.FAILED, NodeStatus.SHUTDOWN, NodeStatus.MAINTENANCE):
        if node.temperature >= 92:
            node.status = NodeStatus.OVERHEATED
        elif node.temperature >= 80:
            node.status = NodeStatus.WARNING
        else:
            node.status = NodeStatus.HEALTHY


# ---------------------------------------------------------------------------
# Visible alerts
# ---------------------------------------------------------------------------

def refresh_visible_alerts(node: GPUNode) -> None:
    alerts: List[str] = []
    if node.temperature >= 92:
        alerts.append("temp_critical")
    elif node.temperature >= 82:
        alerts.append("temp_warning")
    if node.utilization > 0.9:
        alerts.append("utilisation_high")
    if node.throttled:
        alerts.append("throttled")
    if node.status == NodeStatus.MAINTENANCE:
        alerts.append("maintenance")
    if node.status == NodeStatus.SHUTDOWN:
        alerts.append("shutdown")
    if node.maintenance_timer > 0 and node.status != NodeStatus.MAINTENANCE:
        alerts.append("maintenance_pending")
    if node.hidden_degradation > 0.7 and node.temperature > 82:
        # Surface a hint, but still keep the true value hidden.
        alerts.append("latency_spike")
    node.visible_alerts = alerts


def _clamp_temp(t: float) -> float:
    return max(ENV_TEMP_FLOOR, min(ENV_TEMP_CEIL, t))


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------

def average_temperature(nodes: List[GPUNode]) -> float:
    if not nodes:
        return 0.0
    return sum(n.temperature for n in nodes) / len(nodes)


def thermal_safety_score(nodes: List[GPUNode]) -> float:
    """1.0 when everything is cool, 0.0 when everything is overheating."""

    if not nodes:
        return 1.0
    over = sum(1 for n in nodes if n.temperature >= 90.0)
    warn = sum(1 for n in nodes if 80.0 <= n.temperature < 90.0)
    cool = sum(1 for n in nodes if n.temperature < 80.0)
    total = len(nodes)
    return max(0.0, (cool * 1.0 + warn * 0.5 + over * 0.0) / total)
