import json
import logging
import math
import re

from django.conf import settings
from django.shortcuts import render
from django.views.generic import TemplateView
from openai import OpenAI

logger = logging.getLogger(__name__)

client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
    timeout=45.0,
    max_retries=2,
)


SYSTEM_PROMPT = """
You are CircuitPilot AI, an engineering assistant that converts a user's
physical device idea into a structured 3D assembly plan.

Your output will be consumed by a Python backend and a JavaScript Three.js
renderer. The renderer uses your JSON to create 3D components and animate
their assembly.

Return a single valid JSON object only. Do not use Markdown fences,
headings, comments, or text outside the JSON object.

Do not claim that a conceptual design has been physically, electrically,
or mechanically validated. Use approximate dimensions when exact
specifications are unavailable, and disclose assumptions and limitations.

Use this exact top-level structure:

{
  "title": "Device name",
  "description": "Description of the conceptual model",
  "units": "meters",
  "coordinate_system": "right_handed_y_up",
  "camera": {
    "position": [1.3, 1.5, 2.2],
    "target": [0, 1.1, 0]
  },
  "components": [],
  "assembly_steps": [],
  "assumptions": [],
  "validation_warnings": [],
  "limitations": []
}

Each component must contain:
- id: unique lowercase identifier using letters, numbers, and underscores
- name: human-readable component name
- geometry: box, cylinder, sphere, cone, or torus
- dimensions: exactly three positive numbers
- position: [x, y, z] in meters
- rotation: [x, y, z] in radians
- color: six-digit hexadecimal color, such as #3366CC
- material: standard
- opacity: number from 0 to 1
- assembly_step: integer identifying the step where it is introduced
- function: what the component does
- dimensions_verified: boolean
- attached_to: null or the ID of another component

Geometry conventions:
- box dimensions: [width, height, depth]
- cylinder dimensions: [radius, height, radius]; cylinder axis is Y
- sphere dimensions: [radius, radius, radius]
- cone dimensions: [radius, height, radius]; cone axis is Y
- torus dimensions: [major_radius, tube_radius, major_radius]; torus lies in the XZ plane by default

Each assembly step must contain:
- step: consecutive integer starting at 1
- title: short step title
- description: explanation of the assembly action
- component_ids: IDs of the components introduced in this step
- animations: list of animation instructions (one for every component in the step)
- engineering_notes: list of relevant engineering or safety notes

Each animation must contain:
- component_id: existing component ID
- type: fade_in, move_into_place, rotate_into_place, or drop_into_place
- from_position: [x, y, z]
- to_position: [x, y, z]
- from_rotation: [x, y, z]
- to_rotation: [x, y, z]
- duration_ms: integer between 100 and 10000

Every animation destination must match the component's final position
and rotation. Use finite numbers and valid component references.

Each component must be introduced exactly once, in the step matching
its assembly_step. Step numbers must be consecutive, starting at 1.
Use a logical assembly sequence and meaningful final positions.

The camera must contain position and target arrays of three finite
numbers each.

Use arrays of strings for assumptions, validation_warnings, and
limitations. Include relevant uncertainties and safety warnings.
Do not invent verified dimensions, specifications, connections, or tests.

Create a simplified conceptual model for complex devices rather than
inventing large amounts of inaccurate detail.

Placement rules (the model is rendered exactly as you place it):
- Y is up and the floor is y = 0. A part resting on the floor has position.y = height / 2.
- A part stacked on a parent has y = (parent.y + parent_height / 2) + part_height / 2.
  Example: base box [0.4, 0.02, 0.2] at y = 0.01; a battery box [0.06, 0.03, 0.04] on it at y = 0.035.
- Compute positions from dimensions instead of estimating. Touching parts must touch or overlap; gaps over 5 mm are errors.
- Wires are thin boxes that start inside one component's terminal and end inside the next. Their length must span the gap.
- Circuits: mount everything on a base board, put the main parts in a row along X, and wire them in series (battery to resistor to LED and back).
- A cylinder lying along the X axis needs rotation [0, 0, 1.5708].
- Set attached_to on every part except the base.

Before returning the JSON, verify:
- All required fields exist.
- Component IDs are unique.
- Geometry names are supported.
- Dimensions are positive and contain exactly three numbers.
- Positions and rotations contain exactly three finite numbers.
- Colors and opacity values are valid.
- Assembly steps and component references are consistent.
- Animation types, durations, and destinations are valid.
- All output is valid JSON with no surrounding text.

Return the JSON object now.
"""


class LandingPageView(TemplateView):
    template_name = 'cpai/landing.html'


def validate_assembly_plan(data):
    """Perform basic structural validation before passing data to the UI."""

    if not isinstance(data, dict):
        raise ValueError("The AI response must be a JSON object.")

    required_fields = {
        "title",
        "description",
        "units",
        "coordinate_system",
        "camera",
        "components",
        "assembly_steps",
        "assumptions",
        "validation_warnings",
        "limitations",
    }

    if not required_fields.issubset(data.keys()):
        raise ValueError("The assembly plan is missing required fields.")

    if data["units"] != "meters":
        raise ValueError("The assembly plan must use meters.")

    if data["coordinate_system"] != "right_handed_y_up":
        raise ValueError("Unsupported coordinate system.")

    components = data["components"]
    steps = data["assembly_steps"]

    if not isinstance(components, list) or not components:
        raise ValueError("The plan must contain at least one component.")

    if not isinstance(steps, list) or not steps:
        raise ValueError("The plan must contain at least one assembly step.")

    supported_geometry = {"box", "cylinder", "sphere", "cone", "torus"}
    supported_animations = {
        "fade_in",
        "move_into_place",
        "rotate_into_place",
        "drop_into_place",
    }

    def valid_vector(value):
        return (
            isinstance(value, list)
            and len(value) == 3
            and all(
                isinstance(n, (int, float))
                and not isinstance(n, bool)
                and math.isfinite(n)
                for n in value
            )
        )

    component_ids = set()

    for component in components:
        if not isinstance(component, dict):
            raise ValueError("Every component must be an object.")

        component_id = component.get("id")

        if not isinstance(component_id, str) or not component_id:
            raise ValueError("A component has an invalid ID.")

        if component_id in component_ids:
            raise ValueError(f"Duplicate component ID: {component_id}")

        component_ids.add(component_id)

        if component.get("geometry") not in supported_geometry:
            raise ValueError(f"Unsupported geometry for {component_id}.")

        dimensions = component.get("dimensions")

        if not valid_vector(dimensions) or any(d <= 0 for d in dimensions):
            raise ValueError(f"Invalid dimensions for {component_id}.")

        for field in ("position", "rotation"):
            if not valid_vector(component.get(field)):
                raise ValueError(f"Invalid {field} for {component_id}.")

        color = component.get("color", "")
        if not isinstance(color, str) or not re.fullmatch(
            r"#[0-9a-fA-F]{6}", color
        ):
            raise ValueError(f"Invalid color for {component_id}.")

        opacity = component.get("opacity")
        if (
            not isinstance(opacity, (int, float))
            or isinstance(opacity, bool)
            or not 0 <= opacity <= 1
        ):
            raise ValueError(f"Invalid opacity for {component_id}.")

        attached_to = component.get("attached_to")
        if attached_to is not None and not isinstance(attached_to, str):
            raise ValueError(f"Invalid attached_to for {component_id}.")

    for component in components:
        attached_to = component.get("attached_to")
        if attached_to is not None and attached_to not in component_ids:
            raise ValueError(f"Unknown parent component {attached_to}.")

    step_numbers = [step.get("step") for step in steps]

    if step_numbers != list(range(1, len(steps) + 1)):
        raise ValueError("Assembly step numbers must be consecutive.")

    introduced_ids = []

    for step in steps:
        if not isinstance(step.get("component_ids"), list):
            raise ValueError("component_ids must be a list.")

        for component_id in step["component_ids"]:
            if component_id not in component_ids:
                raise ValueError(
                    f"Unknown component in assembly step: {component_id}"
                )
            introduced_ids.append(component_id)

        for animation in step.get("animations", []):
            if animation.get("component_id") not in component_ids:
                raise ValueError("Animation references an unknown component.")

            if animation.get("type") not in supported_animations:
                raise ValueError("Unsupported animation type.")

            duration = animation.get("duration_ms")
            if (
                not isinstance(duration, (int, float))
                or isinstance(duration, bool)
                or not 100 <= duration <= 10000
            ):
                raise ValueError("Invalid animation duration.")

            for field in (
                "from_position",
                "to_position",
                "from_rotation",
                "to_rotation",
            ):
                if not valid_vector(animation.get(field)):
                    raise ValueError(f"Invalid animation field: {field}")

    if len(introduced_ids) != len(set(introduced_ids)):
        raise ValueError("A component is introduced more than once.")

    if set(introduced_ids) != component_ids:
        raise ValueError("Some components are missing from the assembly steps.")

    step_by_number = {step["step"]: step for step in steps}

    for component in components:
        step_number = component.get("assembly_step")

        if (
            not isinstance(step_number, int)
            or isinstance(step_number, bool)
            or step_number not in step_by_number
        ):
            raise ValueError("A component has an invalid assembly_step.")

        if component["id"] not in step_by_number[step_number]["component_ids"]:
            raise ValueError(
                f"assembly_step mismatch for component {component['id']}."
            )

    camera = data["camera"]
    if not isinstance(camera, dict):
        raise ValueError("Camera must be an object.")

    for field in ("position", "target"):
        if not valid_vector(camera.get(field)):
            raise ValueError(f"Invalid camera {field}.")

    for field in ("assumptions", "validation_warnings", "limitations"):
        if (
            not isinstance(data[field], list)
            or not all(isinstance(item, str) for item in data[field])
        ):
            raise ValueError(f"{field} must be a list of strings.")

    return data


def _half_extents(component):
    """Half-size of the component's axis-aligned bounding box."""
    a, b, d = component["dimensions"]
    geometry = component["geometry"]

    if geometry == "box":
        h = [a / 2, b / 2, d / 2]
    elif geometry == "sphere":
        h = [a, a, a]
    elif geometry == "torus":
        h = [a + b, b, a + b]
    else:  # cylinder, cone
        h = [a, b / 2, a]

    rx, ry, rz = component["rotation"]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    # Rotation matrix for three.js "XYZ" Euler order.
    r = [
        [cy * cz, -cy * sz, sy],
        [cx * sz + sx * sy * cz, cx * cz - sx * sy * sz, -sx * cy],
        [sx * sz - cx * sy * cz, sx * cz + cx * sy * sz, cx * cy],
    ]

    return [sum(abs(r[i][j]) * h[j] for j in range(3)) for i in range(3)]


def check_layout(plan, tol=0.01):
    """Return a list of placement problems (floating or disconnected parts)."""
    boxes = {}

    for component in plan["components"]:
        half = _half_extents(component)
        boxes[component["id"]] = [
            (p - e, p + e) for p, e in zip(component["position"], half)
        ]

    problems = []

    if min(box[1][0] for box in boxes.values()) > 2 * tol:
        problems.append(
            "No part rests on the floor (lowest point should be y = 0)."
        )

    for component in plan["components"]:
        parent = component.get("attached_to")
        if not parent:
            continue

        gap = max(
            max(lo1 - hi2, lo2 - hi1, 0)
            for (lo1, hi1), (lo2, hi2) in zip(boxes[component["id"]], boxes[parent])
        )

        if gap > tol:
            problems.append(
                f"{component['name']} floats {gap * 1000:.0f} mm away from "
                f"{parent}; move it so it touches."
            )

    return problems


def generate_plan(question):
    """Ask the model for a plan; retry once if parts float or don't connect."""
    base_input = (
        "Generate the device assembly plan for the following request. "
        "Return only a valid JSON object.\n\n"
        f"Device request: {question}"
    )

    prompt = base_input
    plan = None
    problems = []

    for _ in range(2):
        ai_res = client.responses.create(
            model="gpt-5.6-luna",
            instructions=SYSTEM_PROMPT,
            input=prompt,
            text={"format": {"type": "json_object"}},
        )

        if not ai_res.output_text:
            raise ValueError("The AI returned an empty response.")

        plan = validate_assembly_plan(json.loads(ai_res.output_text))
        problems = check_layout(plan)

        if not problems:
            return plan

        prompt = (
            base_input
            + "\n\nYour previous plan had these placement problems. "
            "Return the full corrected JSON:\n- "
            + "\n- ".join(problems)
        )

    # Still imperfect after the retry: show the issues honestly.
    plan["validation_warnings"].extend(problems)
    return plan


def main_func_temp(request):
    question = ""
    assembly_plan = None
    error = None

    if request.method == "POST":
        question = request.POST.get("question", "").strip()

        if not question:
            error = "Describe the device you want to build."
        elif len(question) > 5000:
            error = "Please keep your device description under 5,000 characters."
        else:
            try:
                assembly_plan = generate_plan(question)

            except (json.JSONDecodeError, ValueError) as exc:
                logger.warning("Invalid CircuitPilot assembly plan: %s", exc)
                error = (
                    "The AI returned an invalid assembly plan. "
                    "Try describing a simpler device."
                )

            except Exception:
                logger.exception("CircuitPilot generation failed.")
                error = (
                    "Something went wrong while generating the device. "
                    "Please try again."
                )

    return render(
        request,
        "cpai/main.html",
        {
            "question": question,
            "assembly_plan": assembly_plan,
            "error": error,
        },
    )