"""Scenario factory.

Each of the eight PRD §16 scenarios is a function that, given a seeded RNG and
a curriculum level, returns a fully populated :class:`ClusterMindState`. The
scenarios encode *initial conditions and arrival schedules* only — the
collapse / rescue dynamics must emerge from the simulator. We never script
"force_collapse" or hardcode which agent will succeed.

Anti-hardcoding rules from PRD §32 are enforced by:
  * randomly selecting which zone or node is degraded per seed,
  * randomly drawing job IDs from a pool,
  * varying arrival timings within a window,
  * letting failure probability come from the dynamics in failures.py.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from clustermind.models import (
    CoolingZone,
    GPUNode,
    IntensityLevel,
    Job,
    JobPriority,
    JobType,
    JobStatus,
    NodeStatus,
    ScenarioConfig,
    ZoneStatus,
)

# ---------------------------------------------------------------------------
# Curriculum descriptors
# ---------------------------------------------------------------------------

CURRICULUM_LEVELS = {
    1: {
        "name": "Scheduling Basics",
        "thermal": False,
        "degradation": False,
        "cascades": False,
        "chaos": False,
        "energy_budget": 2.0,
    },
    2: {
        "name": "Thermal Control",
        "thermal": True,
        "degradation": False,
        "cascades": False,
        "chaos": False,
        "energy_budget": 1.5,
    },
    3: {
        "name": "Hidden Degradation",
        "thermal": True,
        "degradation": True,
        "cascades": False,
        "chaos": False,
        "energy_budget": 1.2,
    },
    4: {
        "name": "Cascading Failures",
        "thermal": True,
        "degradation": True,
        "cascades": True,
        "chaos": False,
        "energy_budget": 1.0,
    },
    5: {
        "name": "Chaos Arena",
        "thermal": True,
        "degradation": True,
        "cascades": True,
        "chaos": True,
        "energy_budget": 1.0,
    },
}


SCENARIO_NAMES = [
    "demand_spike",
    "cooling_failure",
    "hidden_degradation",
    "cascading_failure",
    "energy_squeeze",
    "vip_job_arrival",
    "triple_crisis",
    "chaos_arena",
]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _make_default_nodes(rng: random.Random, n_nodes: int = 10, n_zones: int = 2) -> List[GPUNode]:
    nodes: List[GPUNode] = []
    per_zone = n_nodes // n_zones
    for i in range(n_nodes):
        zone_idx = i // per_zone
        zone_id = f"zone_{chr(ord('A') + zone_idx)}"
        nodes.append(
            GPUNode(
                node_id=f"gpu_{i}",
                zone_id=zone_id,
                total_gpus=8,
                allocated_gpus=0,
                temperature=float(40 + rng.randint(0, 8)),
                hidden_degradation=rng.uniform(0.0, 0.05),
            )
        )
    # Wire neighbor links inside a zone (ring topology per zone for cascades).
    by_zone: Dict[str, List[GPUNode]] = {}
    for n in nodes:
        by_zone.setdefault(n.zone_id, []).append(n)
    for zone_nodes in by_zone.values():
        for idx, node in enumerate(zone_nodes):
            left = zone_nodes[(idx - 1) % len(zone_nodes)]
            right = zone_nodes[(idx + 1) % len(zone_nodes)]
            node.neighbors = [left.node_id, right.node_id]
    return nodes


def _make_default_zones(nodes: List[GPUNode]) -> List[CoolingZone]:
    by_zone: Dict[str, List[str]] = {}
    for n in nodes:
        by_zone.setdefault(n.zone_id, []).append(n.node_id)
    zones = []
    for zone_id, node_ids in sorted(by_zone.items()):
        zones.append(
            CoolingZone(
                zone_id=zone_id,
                cooling_power=1.0,
                cooling_efficiency=0.85,
                cooling_stress=0.0,
                energy_cost_multiplier=1.0,
                node_ids=node_ids,
                intensity=IntensityLevel.LOW,
            )
        )
    return zones


_JOB_POOL_TYPES = [JobType.TRAINING, JobType.INFERENCE, JobType.EVALUATION, JobType.BATCH]


def _make_job(
    rng: random.Random,
    job_id: str,
    priority: JobPriority,
    arrived_at: int = 0,
    gpu_required: Optional[int] = None,
    deadline: Optional[int] = None,
    remaining_work: Optional[float] = None,
    reward_value: Optional[float] = None,
    job_type: Optional[JobType] = None,
) -> Job:
    return Job(
        job_id=job_id,
        job_type=job_type or rng.choice(_JOB_POOL_TYPES),
        priority=priority,
        gpu_required=gpu_required if gpu_required is not None else rng.randint(2, 6),
        deadline_remaining=deadline if deadline is not None else rng.randint(6, 18),
        remaining_work=remaining_work if remaining_work is not None else float(rng.randint(40, 90)),
        thermal_load=rng.uniform(0.8, 1.3),
        energy_cost=rng.uniform(0.02, 0.05),
        reward_value=reward_value if reward_value is not None else _reward_for_priority(priority),
        arrived_at_step=arrived_at,
    )


def _reward_for_priority(priority: JobPriority) -> float:
    return {
        JobPriority.LOW: 0.3,
        JobPriority.MEDIUM: 0.6,
        JobPriority.HIGH: 0.85,
        JobPriority.CRITICAL: 1.0,
    }[priority]


# ---------------------------------------------------------------------------
# Scenario factory
# ---------------------------------------------------------------------------

def build_scenario(
    name: str,
    curriculum_level: int = 1,
    seed: Optional[int] = None,
    n_nodes: int = 10,
    n_zones: int = 2,
    max_steps: int = 20,
) -> Tuple[ScenarioConfig, List[GPUNode], List[CoolingZone], List[Job], List[Dict[str, Any]]]:
    """Return (config, nodes, zones, initial_jobs, arrival_schedule)."""

    if name not in SCENARIO_NAMES:
        raise ValueError(f"Unknown scenario: {name}. Choose from {SCENARIO_NAMES}.")
    rng = random.Random(seed)
    curriculum = CURRICULUM_LEVELS.get(curriculum_level, CURRICULUM_LEVELS[1])

    nodes = _make_default_nodes(rng, n_nodes=n_nodes, n_zones=n_zones)
    zones = _make_default_zones(nodes)

    config = ScenarioConfig(
        name=name,
        curriculum_level=curriculum_level,
        description=_DESCRIPTIONS[name],
        n_nodes=n_nodes,
        n_zones=n_zones,
        max_steps=max_steps,
        chaos_enabled=bool(curriculum["chaos"]),
        chaos_severity_multiplier=0.4 + 0.15 * curriculum_level,
        energy_budget=float(curriculum["energy_budget"]),
        energy_budget_per_step=0.10,
    )

    builder = _SCENARIO_BUILDERS[name]
    initial_jobs, arrival_schedule = builder(rng, nodes, zones, config, curriculum_level)
    return config, nodes, zones, initial_jobs, arrival_schedule


# ---------------------------------------------------------------------------
# Per-scenario builders
# ---------------------------------------------------------------------------

def _scenario_demand_spike(
    rng: random.Random, nodes, zones, config: ScenarioConfig, level: int
) -> Tuple[List[Job], List[Dict[str, Any]]]:
    config.notes.append("Extra jobs flood the queue mid-episode.")
    initial = [
        _make_job(rng, f"job_{i}", _balanced_priority(rng, level)) for i in range(4)
    ]
    spike_step = rng.choice([4, 5, 6])
    burst: List[Job] = []
    for k in range(6):
        priority = rng.choices(
            [JobPriority.HIGH, JobPriority.CRITICAL, JobPriority.MEDIUM],
            weights=[0.4, 0.3, 0.3],
        )[0]
        burst.append(_make_job(rng, f"job_burst_{k}", priority, arrived_at=spike_step))
    return initial, [{"step": spike_step, "jobs": burst}]


def _scenario_cooling_failure(
    rng: random.Random, nodes, zones, config: ScenarioConfig, level: int
) -> Tuple[List[Job], List[Dict[str, Any]]]:
    target_zone = rng.choice(zones)
    config.initial_zone_overrides[target_zone.zone_id] = {"cooling_efficiency": 0.55}
    target_zone.cooling_efficiency = 0.55
    target_zone.status = ZoneStatus.DEGRADED
    config.notes.append(f"{target_zone.zone_id} starts with 55% cooling efficiency.")
    initial = [_make_job(rng, f"job_{i}", _balanced_priority(rng, level)) for i in range(5)]
    return initial, []


def _scenario_hidden_degradation(
    rng: random.Random, nodes, zones, config: ScenarioConfig, level: int
) -> Tuple[List[Job], List[Dict[str, Any]]]:
    chosen = rng.sample(nodes, k=2)
    for node in chosen:
        node.hidden_degradation = rng.uniform(0.45, 0.65)
        config.initial_node_overrides[node.node_id] = {
            "hidden_degradation": node.hidden_degradation,
        }
    config.notes.append(
        "Two random nodes have high hidden degradation; only inspection or "
        "deadline pressure exposes them."
    )
    initial = [_make_job(rng, f"job_{i}", _balanced_priority(rng, level)) for i in range(5)]
    return initial, []


def _scenario_cascading_failure(
    rng: random.Random, nodes, zones, config: ScenarioConfig, level: int
) -> Tuple[List[Job], List[Dict[str, Any]]]:
    seed_node = rng.choice(nodes)
    seed_node.hidden_degradation = rng.uniform(0.55, 0.75)
    seed_node.temperature = rng.uniform(75, 82)
    config.initial_node_overrides[seed_node.node_id] = {
        "hidden_degradation": seed_node.hidden_degradation,
        "temperature": seed_node.temperature,
    }
    config.notes.append(
        f"{seed_node.node_id} starts at high failure risk; neighbours can cascade."
    )
    initial = [_make_job(rng, f"job_{i}", _balanced_priority(rng, level)) for i in range(6)]
    initial[0].priority = JobPriority.CRITICAL
    initial[0].reward_value = _reward_for_priority(JobPriority.CRITICAL)
    return initial, []


def _scenario_energy_squeeze(
    rng: random.Random, nodes, zones, config: ScenarioConfig, level: int
) -> Tuple[List[Job], List[Dict[str, Any]]]:
    config.energy_budget = max(0.6, config.energy_budget * 0.55)
    config.energy_budget_per_step = 0.06
    config.notes.append("Energy budget cut ~45%; cooling/maintenance cost matters.")
    initial = [_make_job(rng, f"job_{i}", _balanced_priority(rng, level)) for i in range(5)]
    return initial, []


def _scenario_vip_job_arrival(
    rng: random.Random, nodes, zones, config: ScenarioConfig, level: int
) -> Tuple[List[Job], List[Dict[str, Any]]]:
    initial = [_make_job(rng, f"job_{i}", JobPriority.MEDIUM) for i in range(5)]
    vip_step = rng.choice([6, 7, 8])
    vip = _make_job(
        rng,
        "job_vip",
        JobPriority.CRITICAL,
        arrived_at=vip_step,
        gpu_required=rng.randint(4, 6),
        deadline=rng.randint(5, 7),
        remaining_work=70.0,
        reward_value=1.2,
        job_type=JobType.INFERENCE,
    )
    config.notes.append(f"VIP job arrives at step {vip_step} with tight deadline.")
    return initial, [{"step": vip_step, "jobs": [vip]}]


def _scenario_triple_crisis(
    rng: random.Random, nodes, zones, config: ScenarioConfig, level: int
) -> Tuple[List[Job], List[Dict[str, Any]]]:
    target_zone = rng.choice(zones)
    target_zone.cooling_efficiency = 0.5
    target_zone.status = ZoneStatus.DEGRADED
    config.initial_zone_overrides[target_zone.zone_id] = {"cooling_efficiency": 0.5}
    chosen_nodes = rng.sample(
        [n for n in nodes if n.zone_id == target_zone.zone_id], k=3
    )
    for node in chosen_nodes:
        node.hidden_degradation = rng.uniform(0.55, 0.7)
        node.temperature = rng.uniform(74, 82)
        config.initial_node_overrides[node.node_id] = {
            "hidden_degradation": node.hidden_degradation,
            "temperature": node.temperature,
        }
    initial = [_make_job(rng, f"job_{i}", _balanced_priority(rng, level)) for i in range(5)]
    spike_step = rng.choice([4, 5])
    burst = [
        _make_job(
            rng,
            f"job_crisis_{k}",
            rng.choice([JobPriority.HIGH, JobPriority.CRITICAL]),
            arrived_at=spike_step,
        )
        for k in range(5)
    ]
    config.notes.append(
        "Combines demand spike, cooling failure, and hidden degradation."
    )
    return initial, [{"step": spike_step, "jobs": burst}]


def _scenario_chaos_arena(
    rng: random.Random, nodes, zones, config: ScenarioConfig, level: int
) -> Tuple[List[Job], List[Dict[str, Any]]]:
    config.chaos_enabled = True
    config.notes.append(
        "Chaos agent is active; bounded events emerge dynamically based on state."
    )
    initial = [_make_job(rng, f"job_{i}", _balanced_priority(rng, level)) for i in range(5)]
    return initial, []


_SCENARIO_BUILDERS = {
    "demand_spike": _scenario_demand_spike,
    "cooling_failure": _scenario_cooling_failure,
    "hidden_degradation": _scenario_hidden_degradation,
    "cascading_failure": _scenario_cascading_failure,
    "energy_squeeze": _scenario_energy_squeeze,
    "vip_job_arrival": _scenario_vip_job_arrival,
    "triple_crisis": _scenario_triple_crisis,
    "chaos_arena": _scenario_chaos_arena,
}


_DESCRIPTIONS = {
    "demand_spike": "Teach priority scheduling under sudden workload pressure.",
    "cooling_failure": "Teach thermal control and cooling tradeoffs.",
    "hidden_degradation": "Teach partial observability via inspection/maintenance.",
    "cascading_failure": "Teach long-horizon failure containment.",
    "energy_squeeze": "Teach energy discipline.",
    "vip_job_arrival": "Teach future priority planning.",
    "triple_crisis": "Combined scenario for storytelling demos.",
    "chaos_arena": "Robustness stress test against adversarial chaos events.",
}


def _balanced_priority(rng: random.Random, level: int) -> JobPriority:
    weights = [0.30, 0.30, 0.25, 0.15] if level <= 2 else [0.20, 0.30, 0.30, 0.20]
    return rng.choices(
        [JobPriority.LOW, JobPriority.MEDIUM, JobPriority.HIGH, JobPriority.CRITICAL],
        weights=weights,
    )[0]


def list_scenarios() -> List[str]:
    return list(SCENARIO_NAMES)


def describe(name: str) -> str:
    return _DESCRIPTIONS.get(name, "")
