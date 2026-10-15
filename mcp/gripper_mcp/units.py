"""Conversions between the agent-facing opening (mm) and the controller's joint angle.

Two domains are in play and they must never be confused:

- **opening_mm**: what agents and tools speak. 0.0 = closed, the model's max
  opening (85.0 on a 2F-85) = fully open.
- **knuckle_rad**: what `robotiq_gripper_controller` speaks, the position of the
  gripper's command joint. 0.0 = open, the driver's closed position
  (`gripper_closed_position` in the robot description, 0.7929 on a 2F-85)
  = closed. INVERTED relative to opening_mm.

The two ends have two owners. The stroke in mm is the product datasheet's and
ships with the server. The command joint's name and range belong to the robot
description the backend is actually talking to, so the backend supplies them at
runtime and nothing here is copied from a URDF.

The mapping between them is linear. The real linkage is a four-bar, so
mid-stroke openings are approximate; the ends are exact only as far as the
backend's closed angle is the one the driver actually closes to.

This module is a stopgap. The driver already maps the joint angle to register
counts in C++ (robotiq_driver's gripper_scaling.hpp), and the counts-to-mm
conversion belongs with it, in the SDK, so that every consumer shares one
calibration (robotiq/grippers#16). Until that exists the MCP can only reach the
joint angle over ROS, so it carries this small second map.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Stroke:
    max_opening_mm: float
    closed_tolerance_mm: float
    min_opening_mm: float = 0.0

    def __post_init__(self) -> None:
        if self.max_opening_mm <= self.min_opening_mm:
            raise ValueError("max_opening_mm must exceed min_opening_mm")
        if (
            not 0.0
            < self.closed_tolerance_mm
            < self.max_opening_mm - self.min_opening_mm
        ):
            raise ValueError(
                "closed_tolerance_mm must be positive and inside the stroke"
            )

    def fingers_met(self, opening_mm: float) -> bool:
        return opening_mm <= self.min_opening_mm + self.closed_tolerance_mm

    def fingers_open(self, opening_mm: float) -> bool:
        return opening_mm >= self.max_opening_mm - self.closed_tolerance_mm


@dataclass(frozen=True)
class JointGeometry:
    name: str
    rad_open: float
    rad_closed: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("the command joint needs a name")
        if self.rad_closed == self.rad_open:
            raise ValueError("rad_closed must differ from rad_open")


@dataclass(frozen=True)
class GripperGeometry:
    stroke: Stroke
    joint: JointGeometry


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clamp_opening_mm(opening_mm: float, geometry: GripperGeometry) -> float:
    stroke = geometry.stroke
    return clamp(opening_mm, stroke.min_opening_mm, stroke.max_opening_mm)


def opening_mm_to_fraction(opening_mm: float, geometry: GripperGeometry) -> float:
    stroke = geometry.stroke
    span = stroke.max_opening_mm - stroke.min_opening_mm
    return (clamp_opening_mm(opening_mm, geometry) - stroke.min_opening_mm) / span


def opening_mm_to_knuckle_rad(opening_mm: float, geometry: GripperGeometry) -> float:
    fraction = opening_mm_to_fraction(opening_mm, geometry)
    joint = geometry.joint
    return joint.rad_closed - fraction * (joint.rad_closed - joint.rad_open)


def knuckle_rad_to_opening_mm(knuckle_rad: float, geometry: GripperGeometry) -> float:
    joint, stroke = geometry.joint, geometry.stroke
    fraction = (joint.rad_closed - knuckle_rad) / (joint.rad_closed - joint.rad_open)
    opening_span = stroke.max_opening_mm - stroke.min_opening_mm
    return stroke.min_opening_mm + clamp(fraction, 0.0, 1.0) * opening_span
