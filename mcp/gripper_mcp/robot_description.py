"""Read the gripper's command joint and driver scale from a robot description, with no ROS imports.

A Robotiq 2F is one actuated joint plus finger joints that `<mimic>` it, so the
command joint is the one the others point at. That holds whatever `prefix` the
xacro was expanded with, which is why the name is read here rather than kept in
a datasheet. The joint's lower limit is the open angle.

The rest is the driver's, carried as `<hardware>` params of the `<ros2_control>`
block that lists the joint. The closed angle is `gripper_closed_position`
(0.7929 on a 2F-85, against a limit of 0.8), the angle the driver scales the
position register to. `gripper_max_force` and `gripper_max_speed` are what the
driver divides a goal's effort and speed by to fill the force and speed
registers (235 N and 0.150 m/s on a 2F-85, 125 N and 0.250 m/s on a 2F-140, and
overridable per cell), so a fraction of the gripper's range is sent as that
fraction of them. Where the block has no such param (the fake hardware, the
topic-based sim) the closed angle falls back to the joint limit and the scale to
the driver's own defaults, which is what a driver started without them would use.
"""

from dataclasses import dataclass
from xml.etree import ElementTree

from gripper_mcp.units import JointGeometry

DRIVER_DEFAULT_MAX_FORCE_N = 235.0
DRIVER_DEFAULT_MAX_SPEED_M_S = 0.150


@dataclass(frozen=True)
class DriverScale:
    max_force_n: float
    max_speed_m_s: float


def command_joint(urdf_xml: str) -> JointGeometry:
    robot = parse(urdf_xml)
    joints = joints_by_name(robot)
    name = mimicked_joint(robot, set(joints))
    rad_open, limit_upper = limits(joints[name])
    closed = hardware_param(robot, name, "gripper_closed_position", limit_upper)
    return JointGeometry(name, rad_open, closed)


def driver_scale(urdf_xml: str) -> DriverScale:
    robot = parse(urdf_xml)
    name = mimicked_joint(robot, set(joints_by_name(robot)))
    return DriverScale(
        max_force_n=hardware_param(
            robot, name, "gripper_max_force", DRIVER_DEFAULT_MAX_FORCE_N
        ),
        max_speed_m_s=hardware_param(
            robot, name, "gripper_max_speed", DRIVER_DEFAULT_MAX_SPEED_M_S
        ),
    )


def parse(urdf_xml: str) -> ElementTree.Element:
    try:
        return ElementTree.fromstring(urdf_xml)
    except ElementTree.ParseError as error:
        raise ValueError(f"robot_description is not valid XML: {error}") from error


def joints_by_name(robot: ElementTree.Element) -> dict[str, ElementTree.Element]:
    joints = {}
    for joint in robot.findall("joint"):
        if "name" not in joint.attrib:
            raise ValueError("robot_description has a <joint> without a name")
        joints[joint.attrib["name"]] = joint
    return joints


def mimicked_joint(robot: ElementTree.Element, joint_names: set[str]) -> str:
    mimicked = sorted(
        {mimic.attrib.get("joint", "") for mimic in robot.iter("mimic")} & joint_names
    )
    if len(mimicked) != 1:
        raise ValueError(
            "Expected exactly one joint mimicked by the finger joints, found "
            f"{len(mimicked)}: {', '.join(mimicked) or '(none)'}. Is this the "
            "description of a single Robotiq 2F gripper?"
        )
    return mimicked[0]


def hardware_param(
    robot: ElementTree.Element, joint: str, param: str, fallback: float
) -> float:
    for block in robot.iter("ros2_control"):
        if block.find(f"joint[@name='{joint}']") is None:
            continue
        found = block.find(f"hardware/param[@name='{param}']")
        if found is not None and found.text:
            return float(found.text)
    return fallback


def limits(joint: ElementTree.Element) -> tuple[float, float]:
    limit = joint.find("limit")
    try:
        return float(limit.attrib["lower"]), float(limit.attrib["upper"])
    except (AttributeError, KeyError) as error:
        raise ValueError(
            f"Joint '{joint.attrib['name']}' has no <limit lower= upper=> in the URDF"
        ) from error
