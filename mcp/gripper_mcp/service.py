"""Translation between the MCP tool surface and the backends.

Backends speak the command joint's angle; tools speak millimetres. Everything in
between, unit conversion, outcome classification, per-gripper dispatch, lives
here so it can be tested without a FastMCP server or a ROS graph.

`classify` does not decide whether a stop was wanted. Every motion reports
`stopped_on_object` neutrally when the fingers halted short of the commanded
position, in either direction, and the caller, who knows the intent, reads it
as a grasp or an obstruction. The direction is visible from the commanded and
achieved openings. The gripper's own firmware answers the same question with
one enum (moving, detected while opening, detected while closing, at the
requested position), which is what this vocabulary mirrors.

The outcome is the controller's verdict, never inferred here: `reached_goal`
means `reached`, `stalled` means `stopped_on_object`, and a goal it decided
neither way is `incomplete`. On a real gripper the controller takes both flags
from the firmware's own object detection (gOBJ, the `object_status` state
interface); the simulators' configs decide them from position and velocity.

A number that is not finite is refused before it reaches a clamp: `max` and
`min` pass NaN through to the upper bound, so a NaN effort would grip at full
force and a NaN opening open fully.

The controller calls a move done within about 2 mm of its goal, before the
fingers settle, so a full close on nothing can read 1.9 mm while the closed
tolerance is 1.5. `fingers_met` therefore also trusts the last full close this
server saw come back `reached`, as long as the opening has stayed within twice
the tolerance; otherwise the TSF pads, pressing on each other, would make an
empty gripper read as a hold.
"""

import math
from datetime import datetime, timezone

from gripper_mcp.backend import BackendMotion, GripperBackend
from gripper_mcp.config import GripperConfig, GripperModelSpec, SpeedRange
from gripper_mcp.models import (
    GripperHealth,
    GripperInfo,
    GripperMotionResult,
    GripperState,
    Outcome,
)
from gripper_mcp.units import (
    GripperGeometry,
    Stroke,
    clamp,
    clamp_opening_mm,
    knuckle_rad_to_opening_mm,
    opening_mm_to_fraction,
    opening_mm_to_knuckle_rad,
)

MIN_EFFORT = 0.0
MAX_EFFORT = 1.0
MM_DECIMALS = 2
RAD_DECIMALS = 4
FRACTION_DECIMALS = 4
SETTLE_TOLERANCES = 2.0


class UnknownGripperError(Exception):
    pass


class GripperService:
    def __init__(
        self,
        specs: dict[str, GripperModelSpec],
        configs: dict[str, GripperConfig],
        backends: dict[str, GripperBackend],
        tactile_sources: dict[str, str] | None = None,
    ) -> None:
        self._specs = specs
        self._configs = configs
        self._backends = backends
        self._tactile_sources = tactile_sources or {}
        self._closed_on_reach: set[str] = set()

    def list_grippers(self) -> list[GripperInfo]:
        return [
            GripperInfo(
                gripper_name=name,
                model=config.model,
                backend=self._backends[name].name,
                max_opening_mm=self._specs[config.model].stroke.max_opening_mm,
                min_speed_mm_s=self._specs[config.model].speed.min_mm_s,
                max_speed_mm_s=self._specs[config.model].speed.max_mm_s,
                tactile=self._tactile_sources.get(name),
                description=config.description,
            )
            for name, config in self._configs.items()
        ]

    def stroke_of(self, gripper_name: str) -> Stroke:
        return self._spec(gripper_name).stroke

    def fingers_met(self, gripper_name: str, opening_mm: float) -> bool:
        stroke = self.stroke_of(gripper_name)
        if stroke.fingers_met(opening_mm):
            return True
        settled_bound = (
            stroke.min_opening_mm + SETTLE_TOLERANCES * stroke.closed_tolerance_mm
        )
        return gripper_name in self._closed_on_reach and opening_mm <= settled_bound

    def get_state(self, gripper_name: str) -> GripperState:
        backend = self._backend(gripper_name)
        geometry = geometry_of(self._spec(gripper_name), backend)
        state = backend.read_state()
        opening_mm = knuckle_rad_to_opening_mm(state.position_rad, geometry)

        return GripperState(
            gripper_name=gripper_name,
            opening_mm=round(opening_mm, MM_DECIMALS),
            opening_fraction=round(
                opening_mm_to_fraction(opening_mm, geometry), FRACTION_DECIMALS
            ),
            knuckle_rad=round(state.position_rad, RAD_DECIMALS),
            holding_effort=state.holding_effort,
            backend=backend.name,
            measured_at=timestamp(),
        )

    def open_fully(
        self,
        gripper_name: str,
        effort: float | None = None,
        speed_mm_s: float | None = None,
    ) -> GripperMotionResult:
        stroke = self._spec(gripper_name).stroke
        return self.move_to_opening(
            gripper_name, stroke.max_opening_mm, effort, speed_mm_s
        )

    def close_fully(
        self,
        gripper_name: str,
        effort: float | None = None,
        speed_mm_s: float | None = None,
    ) -> GripperMotionResult:
        stroke = self._spec(gripper_name).stroke
        return self.move_to_opening(
            gripper_name, stroke.min_opening_mm, effort, speed_mm_s
        )

    def move_to_opening(
        self,
        gripper_name: str,
        opening_mm: float,
        effort: float | None = None,
        speed_mm_s: float | None = None,
    ) -> GripperMotionResult:
        require_finite(position_mm=opening_mm, effort=effort, speed_mm_s=speed_mm_s)
        backend = self._backend(gripper_name)
        spec = self._spec(gripper_name)
        geometry = geometry_of(spec, backend)
        target_mm = clamp_opening_mm(opening_mm, geometry)
        applied_effort = clamp_effort(effort)
        applied_speed = clamp_speed(speed_mm_s, spec.speed)
        motion = backend.move_to(
            position_rad=opening_mm_to_knuckle_rad(target_mm, geometry),
            effort=applied_effort,
            timeout_s=spec.defaults.motion_timeout_s,
            speed=(
                None if applied_speed is None else applied_speed / spec.speed.max_mm_s
            ),
        )
        outcome = classify(motion)
        self._closed_on_reach.discard(gripper_name)
        if target_mm == spec.stroke.min_opening_mm and outcome == "reached":
            self._closed_on_reach.add(gripper_name)

        return GripperMotionResult(
            gripper_name=gripper_name,
            commanded_opening_mm=round(target_mm, MM_DECIMALS),
            achieved_opening_mm=achieved_opening_mm(motion, geometry),
            reached_goal=motion.reached_goal,
            stalled=motion.stalled,
            object_detected=outcome == "stopped_on_object",
            outcome=outcome,
            detail=(
                clamp_note(opening_mm, target_mm)
                + effort_note(effort, applied_effort)
                + speed_note(speed_mm_s, applied_speed)
                + motion.detail
            ),
            backend=backend.name,
        )

    def get_health(self, gripper_name: str) -> GripperHealth:
        backend = self._backend(gripper_name)
        health = backend.health()

        return GripperHealth(
            gripper_name=gripper_name,
            reachable=health.reachable,
            controller_active=health.controller_active,
            detail=health.detail,
            backend=backend.name,
        )

    def assert_known(self, gripper_name: str) -> None:
        if gripper_name not in self._configs:
            available = ", ".join(self._configs) or "(none)"
            raise UnknownGripperError(
                f"Unknown gripper '{gripper_name}'. Available: {available}"
            )

    def _backend(self, gripper_name: str) -> GripperBackend:
        self.assert_known(gripper_name)
        return self._backends[gripper_name]

    def _spec(self, gripper_name: str) -> GripperModelSpec:
        self.assert_known(gripper_name)
        return self._specs[self._configs[gripper_name].model]


def geometry_of(spec: GripperModelSpec, backend: GripperBackend) -> GripperGeometry:
    return GripperGeometry(spec.stroke, backend.joint_geometry())


def timestamp() -> datetime:
    return datetime.now(timezone.utc)


def require_finite(**values: float | None) -> None:
    for name, value in values.items():
        if value is not None and not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number, got {value}.")


def achieved_opening_mm(
    motion: BackendMotion, geometry: GripperGeometry
) -> float | None:
    if motion.refused or motion.final_position_rad is None:
        return None
    opening = knuckle_rad_to_opening_mm(motion.final_position_rad, geometry)
    return round(opening, MM_DECIMALS)


def clamp_note(requested_mm: float, target_mm: float) -> str:
    if requested_mm == target_mm:
        return ""
    return f"Requested {requested_mm:.1f} mm, clamped to {target_mm:.1f} mm. "


def clamp_effort(requested: float | None) -> float | None:
    if requested is None:
        return None
    return clamp(requested, MIN_EFFORT, MAX_EFFORT)


def effort_note(requested: float | None, applied: float | None) -> str:
    if requested is None or requested == applied:
        return ""
    return f"Requested effort {requested:g}, clamped to {applied:g}. "


def clamp_speed(requested: float | None, speed: SpeedRange) -> float | None:
    if requested is None:
        return None
    return clamp(requested, speed.min_mm_s, speed.max_mm_s)


def speed_note(requested: float | None, applied: float | None) -> str:
    if requested is None or requested == applied:
        return ""
    return f"Requested speed {requested:g} mm/s, clamped to {applied:g} mm/s. "


def classify(motion: BackendMotion) -> Outcome:
    if motion.refused:
        return "refused"
    if motion.timed_out:
        return "incomplete"
    if motion.reached_goal:
        return "reached"
    if motion.stalled:
        return "stopped_on_object"
    return "incomplete"
