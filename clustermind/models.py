"""Pydantic data contracts for ClusterMind Chaos Arena.

All cross-module data structures live here so the rest of the package can be
written against stable types. The ``Action`` / ``Observation`` / ``State``
base classes come from ``openenv-core`` when available, otherwise we fall back
to local Pydantic shims with the same fields the OpenEnv runtime expects.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

try:
    from openenv.core.env_server import Action as _OEAction
    from openenv.core.env_server import Observation as _OEObservation
    from openenv.core.env_server import State as _OEState

    _OPENENV_BASES = True
except Exception:  # pragma: no cover - exercised when openenv-core absent
    from pydantic import BaseModel as _BM

    class _OEAction(_BM):
        metadata: Dict[str, Any] = {}

    class _OEObservation(_BM):
        done: bool = False
        reward: Optional[float] = None
        metadata: Dict[str, Any] = {}

    class _OEState(_BM):
        episode_id: Optional[str] = None
        step_count: int = 0

    _OPENENV_BASES = False


from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    ALLOCATE_JOB = "ALLOCATE_JOB"
    DELAY_JOB = "DELAY_JOB"
    THROTTLE_NODE = "THROTTLE_NODE"
    INCREASE_COOLING = "INCREASE_COOLING"
    RUN_MAINTENANCE = "RUN_MAINTENANCE"
    MIGRATE_JOB = "MIGRATE_JOB"
    INSPECT_NODE = "INSPECT_NODE"
    SHUTDOWN_NODE = "SHUTDOWN_NODE"
    NO_OP = "NO_OP"


class ChaosActionType(str, Enum):
    INJECT_DEMAND_SPIKE = "INJECT_DEMAND_SPIKE"
    DROP_COOLING_EFFICIENCY = "DROP_COOLING_EFFICIENCY"
    INCREASE_HIDDEN_DEGRADATION = "INCREASE_HIDDEN_DEGRADATION"
    ADD_VIP_JOB = "ADD_VIP_JOB"
    REDUCE_ENERGY_BUDGET = "REDUCE_ENERGY_BUDGET"
    DELAY_MAINTENANCE = "DELAY_MAINTENANCE"
    TRIGGER_LATENCY_ALERT = "TRIGGER_LATENCY_ALERT"
    NO_CHAOS = "NO_CHAOS"


class JobType(str, Enum):
    TRAINING = "training"
    INFERENCE = "inference"
    EVALUATION = "evaluation"
    BATCH = "batch"


class JobPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DELAYED = "delayed"
    COMPLETED = "completed"
    FAILED = "failed"


class NodeStatus(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    OVERHEATED = "overheated"
    FAILED = "failed"
    MAINTENANCE = "maintenance"
    SHUTDOWN = "shutdown"


class ZoneStatus(str, Enum):
    NORMAL = "normal"
    STRESSED = "stressed"
    DEGRADED = "degraded"


class IntensityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class GuardrailSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


PRIORITY_WEIGHT = {
    JobPriority.LOW: 0.25,
    JobPriority.MEDIUM: 0.50,
    JobPriority.HIGH: 0.75,
    JobPriority.CRITICAL: 1.00,
}

INTENSITY_MULTIPLIER = {
    IntensityLevel.LOW: 0.4,
    IntensityLevel.MEDIUM: 0.7,
    IntensityLevel.HIGH: 1.0,
}


# ---------------------------------------------------------------------------
# Domain entities
# ---------------------------------------------------------------------------

class GPUNode(BaseModel):
    node_id: str
    zone_id: str
    total_gpus: int
    allocated_gpus: int = 0
    temperature: float = 45.0
    utilization: float = 0.0
    status: NodeStatus = NodeStatus.HEALTHY
    assigned_jobs: List[str] = Field(default_factory=list)
    visible_alerts: List[str] = Field(default_factory=list)
    hidden_degradation: float = 0.0
    hidden_failure_probability: float = 0.0
    maintenance_timer: int = 0
    throttled: bool = False
    last_failure_step: Optional[int] = None
    inspection_estimate: Optional[float] = None
    inspection_step: Optional[int] = None
    neighbors: List[str] = Field(default_factory=list)

    @property
    def free_gpus(self) -> int:
        return max(0, self.total_gpus - self.allocated_gpus)

    @property
    def is_active(self) -> bool:
        return self.status not in (NodeStatus.FAILED, NodeStatus.SHUTDOWN, NodeStatus.MAINTENANCE)


class CoolingZone(BaseModel):
    zone_id: str
    cooling_power: float = 1.0
    cooling_efficiency: float = 1.0
    cooling_stress: float = 0.0
    energy_cost_multiplier: float = 1.0
    status: ZoneStatus = ZoneStatus.NORMAL
    node_ids: List[str] = Field(default_factory=list)
    intensity: IntensityLevel = IntensityLevel.MEDIUM
    high_cooling_steps: int = 0


class Job(BaseModel):
    job_id: str
    job_type: JobType
    priority: JobPriority
    gpu_required: int
    deadline_remaining: int
    waiting_steps: int = 0
    remaining_work: float = 100.0
    progress: float = 0.0
    thermal_load: float = 1.0
    energy_cost: float = 0.05
    reward_value: float = 1.0
    status: JobStatus = JobStatus.QUEUED
    assigned_node: Optional[str] = None
    arrived_at_step: int = 0
    completed_at_step: Optional[int] = None
    delay_count: int = 0


# ---------------------------------------------------------------------------
# Action / Observation / State
# ---------------------------------------------------------------------------

class ActionRationale(BaseModel):
    """Optional structured rationale used for demo interpretability.

    Per PRD §23, this MUST NOT influence reward.
    """

    risk_assessment: Optional[str] = None
    selected_strategy: Optional[str] = None
    chosen_action: Optional[str] = None
    expected_effect: Optional[str] = None


class ClusterMindAction(_OEAction):
    """Primary agent action.

    A single typed action covers all 9 verbs from PRD §12. Per-verb required
    fields are enforced at validation time inside the simulator, not here, so
    that malformed inputs land in the invalid-action penalty path rather than
    crashing the server.
    """

    action_type: ActionType = ActionType.NO_OP
    job_id: Optional[str] = None
    node_id: Optional[str] = None
    target_id: Optional[str] = None
    source_node_id: Optional[str] = None
    target_node_id: Optional[str] = None
    zone_id: Optional[str] = None
    intensity: Optional[IntensityLevel] = None
    rationale: Optional[ActionRationale] = None


class NodeView(BaseModel):
    """Partial view of a GPU node exposed to the agent."""

    node_id: str
    zone_id: str
    total_gpus: int
    free_gpus: int
    allocated_gpus: int
    temperature: float
    utilization: float
    status: NodeStatus
    assigned_jobs: List[str]
    visible_alerts: List[str]
    throttled: bool
    maintenance_timer: int
    inspection_estimate: Optional[float] = None


class JobView(BaseModel):
    job_id: str
    job_type: JobType
    priority: JobPriority
    gpu_required: int
    deadline_remaining: int
    waiting_steps: int
    progress_pct: float
    status: JobStatus
    assigned_node: Optional[str]
    reward_value: float


class ZoneView(BaseModel):
    zone_id: str
    cooling_power: float
    cooling_efficiency: float
    cooling_stress: float
    intensity: IntensityLevel
    status: ZoneStatus
    node_ids: List[str]


class LegalActionDescriptor(BaseModel):
    """A short hint about what verb is currently legal somewhere in the cluster.

    Agents should still validate per-target preconditions in their JSON action;
    this list exists primarily to anchor the LLM prompt.
    """

    action_type: ActionType
    description: str


class ClusterMindObservation(_OEObservation):
    step: int = 0
    max_steps: int = 20
    scenario: str = "demand_spike"
    curriculum_level: int = 1

    cluster_health: float = 1.0
    energy_remaining: float = 1.0
    queue_pressure: float = 0.0
    average_temperature: float = 45.0
    active_outages: int = 0
    cascade_count: int = 0

    nodes: List[NodeView] = Field(default_factory=list)
    cooling_zones: List[ZoneView] = Field(default_factory=list)
    jobs: List[JobView] = Field(default_factory=list)
    alerts: List[str] = Field(default_factory=list)
    legal_actions: List[LegalActionDescriptor] = Field(default_factory=list)
    last_action_result: Optional[str] = None
    guardrail_warnings: List[str] = Field(default_factory=list)


class ClusterMindState(_OEState):
    """Hidden debug state — only returned by ``env.state``.

    Per PRD §11 the agent observation must remain partial; anything stored
    here is *not* expected to be exposed during a step.
    """

    scenario: str = "demand_spike"
    curriculum_level: int = 1
    seed: Optional[int] = None
    nodes: List[GPUNode] = Field(default_factory=list)
    cooling_zones: List[CoolingZone] = Field(default_factory=list)
    jobs: List[Job] = Field(default_factory=list)
    energy_remaining: float = 1.0
    energy_budget_per_step: float = 0.10
    cluster_health: float = 1.0
    cascade_count: int = 0
    outage_count: int = 0
    completed_jobs: List[str] = Field(default_factory=list)
    failed_jobs: List[str] = Field(default_factory=list)
    chaos_events: List["ChaosEvent"] = Field(default_factory=list)
    guardrail_violations: List["GuardrailViolation"] = Field(default_factory=list)
    invalid_action_count: int = 0
    no_op_count: int = 0
    pending_chaos_cooldown: int = 0
    last_chaos_action: Optional[ChaosActionType] = None


# ---------------------------------------------------------------------------
# Auxiliary types
# ---------------------------------------------------------------------------

class ChaosEvent(BaseModel):
    step: int
    action: ChaosActionType
    target: Optional[str] = None
    severity: float = 0.5
    detail: str = ""


class GuardrailViolation(BaseModel):
    step: int
    name: str
    severity: GuardrailSeverity
    explanation: str
    penalty: float


class RewardBreakdown(BaseModel):
    critical_job_completion: float = 0.0
    normal_job_completion: float = 0.0
    deadline_score: float = 0.0
    cluster_health_score: float = 0.0
    thermal_safety_score: float = 0.0
    recovery_score: float = 0.0
    energy_efficiency_score: float = 0.0
    useful_inspection_or_maintenance: float = 0.0
    outage_penalty: float = 0.0
    cascade_penalty: float = 0.0
    missed_critical_deadline_penalty: float = 0.0
    guardrail_violation_penalty: float = 0.0
    invalid_action_penalty: float = 0.0
    no_progress_penalty: float = 0.0
    total: float = 0.0


class FlightRecord(BaseModel):
    record_id: str
    step: int
    observation_summary: Dict[str, Any]
    primary_action: Dict[str, Any]
    chaos_action: Optional[str] = None
    reward_breakdown: Dict[str, float]
    guardrail_flags: List[str] = Field(default_factory=list)
    failure_events: List[str] = Field(default_factory=list)
    cascade_events: List[str] = Field(default_factory=list)
    failure_risk: Dict[str, float] = Field(default_factory=dict)
    event_log: List[str] = Field(default_factory=list)


class ScenarioConfig(BaseModel):
    name: str
    curriculum_level: int
    description: str
    n_nodes: int = 10
    n_zones: int = 2
    max_steps: int = 20
    initial_jobs: int = 6
    arrival_schedule: List[Dict[str, Any]] = Field(default_factory=list)
    initial_node_overrides: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    initial_zone_overrides: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    energy_budget: float = 1.0
    energy_budget_per_step: float = 0.10
    chaos_enabled: bool = False
    chaos_severity_multiplier: float = 1.0
    notes: List[str] = Field(default_factory=list)


# Forward refs for State (Pydantic v2 handles automatically when using `from __future__ import annotations`).
ClusterMindState.model_rebuild()
