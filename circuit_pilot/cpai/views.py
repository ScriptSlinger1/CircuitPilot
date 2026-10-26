import json
import logging
import math
import re
import time

from django.conf import settings
from django.shortcuts import render
from django.views.generic import TemplateView
from openai import APITimeoutError, OpenAI

logger = logging.getLogger(__name__)

client = OpenAI(
    api_key=settings.OPENAI_API_KEY,
    timeout=45.0,
    max_retries=2,
)


# Assembly plans are long JSON documents, so one request can take well over the
# client's 45 s default. This overrides the timeout for plan requests only; the
# client above (key, max_retries) is left exactly as you configured it.
AI_REQUEST_TIMEOUT = 120.0   # seconds allowed per request
AI_TOTAL_BUDGET = 180.0      # skip the corrective retry once half of this is used


# Component types the 3D viewer can draw as detailed multi-part models.
# "generic" (and plans that predate component_type) use the primitive renderer.
DETAILED_COMPONENT_TYPES = (
    "resistor",
    "led",
    "capacitor",
    "battery",
    "breadboard",
    "pcb",
    "wire",
    "microcontroller",
    "screw",
    "motor",
    "gear",
)
SUPPORTED_COMPONENT_TYPES = DETAILED_COMPONENT_TYPES + ("generic",)


SYSTEM_PROMPT = """
You are CircuitPilot AI, an engineering assistant that converts a user's
physical device idea into a structured 3D assembly plan.

Your output will be consumed by a Python backend and a JavaScript Three.js
renderer. The renderer uses your JSON to build recognizable 3D models of the
components and animate their assembly.

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
    "position": [0.3, 0.25, 0.4],
    "target": [0, 0.02, 0]
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
- component_type: one of resistor, led, capacitor, battery, breadboard, pcb,
  wire, microcontroller, screw, motor, gear, generic
- geometry: box, cylinder, sphere, cone, or torus
- dimensions: exactly three positive numbers
- position: [x, y, z] in meters (the centre of the part's bounding box)
- rotation: [x, y, z] in radians
- color: six-digit hexadecimal color, such as #3366CC
- material: standard
- opacity: number from 0 to 1
- assembly_step: integer identifying the step where it is introduced
- function: what the component does
- dimensions_verified: boolean (true only if the user gave exact dimensions or
  you are certain of them; false for every estimate)
- attached_to: null or the ID of another component

COMPONENT TYPES
Pick the most specific supported component_type for every part. Use "generic"
only when no listed type fits (brackets, housings, poles, plates, shafts...).
An LED circuit should use breadboard, battery, resistor, led and wire parts.
A motor assembly should use motor, screw and gear parts plus generic mounts.

For every component_type except "generic":
- set geometry to "box";
- dimensions are the part's overall bounding box [x, y, z] in meters: FULL
  lengths, not radii. The renderer builds a detailed model that fills exactly
  that box, so use realistic real-world sizes (do not enlarge small parts; the
  viewer zooms to them);
- the model is drawn automatically along the axis described below, so do not
  add rotation just to make a part lie along an axis; use rotation only to
  turn a part to a different heading;
- color is the part's main colour as listed.

Per-type conventions (typical dimensions in meters):
- resistor: axial resistor including its leads; its longest dimension is its
  axis (X by convention). 1/4 W: [0.030, 0.0025, 0.0025]. color = body colour
  (#D9B27C). If you choose a value, put it in the name (for example
  "330 Ω resistor") and list it under assumptions; the renderer paints the
  colour bands from the value in the name.
- led: 5 mm through-hole LED standing along Y: dome up, leads down. The longer
  anode (+) lead is on local +X, the cathode (-) lead on local -X.
  [0.0058, 0.030, 0.0058]. color = lens colour (for example #FF2A2A).
- capacitor: radial electrolytic can standing along Y, leads down, positive
  lead on +X, negative stripe on -X. [0.0065, 0.020, 0.0065]. color = sleeve.
- battery: the shape is chosen from the proportions of the box. Cylindrical
  cell such as AA, lying along X: [0.0505, 0.0145, 0.0145] (positive nub at
  the + end of the long axis). Rectangular 9 V style block:
  [0.0265, 0.0485, 0.0175] (snap terminals on top +Y; positive stud at +X,
  negative socket at -X). Coin cell: [0.020, 0.0032, 0.020] (positive face
  +Y). color = wrapper / body colour.
- breadboard: [0.165, 0.0085, 0.055] (830 tie points) or half size
  [0.0825, 0.0085, 0.055]. Top surface at y = position.y + height / 2.
  Holes sit on a pitch of depth / 21.65 (0.00254 for a 0.055 deep board).
  Columns run along X one pitch apart, centred on the board. Across Z, the
  ten terminal rows are at +/-1.5, 2.5, 3.5, 4.5, 5.5 pitches from the
  board's centre line (the centre channel is between +/-1.5) and the four
  power-rail rows are at +/-8 and +/-9 pitches. color = plastic (#ECEAE3).
- pcb: bare circuit board with pads, traces and mounting holes on the top
  (+Y) face. [0.100, 0.0016, 0.070]. color = solder mask (#0B6B3A).
- wire: straight insulated wire with bare tips at both ends; its longest
  dimension is its axis (X by convention). [length, 0.002, 0.002]. Make the
  length the distance between the two end points plus about 0.004 so the tips
  sink into the terminals. To run a wire from start S to end E, set position
  to the midpoint and rotation to
  [0, -atan2(dz, dx), atan2(dy, sqrt(dx*dx + dz*dz))] where
  (dx, dy, dz) = E - S. Use several wires for bent runs. color = insulation.
- microcontroller: development board with a USB connector and power jack on
  the -X edge and pin headers along both long edges.
  [0.0686, 0.0135, 0.0534] (height includes the tallest connector).
  color = board colour (#1F6FB5). Do not present it as a specific product.
- screw: stands along Y with the head up. [head_diameter, length, head_diameter],
  for example M3 x 10: [0.0055, 0.012, 0.0055]. color = metal finish.
- motor: small DC motor; its longest dimension is its axis, with the shaft
  at the + end. [0.020, 0.030, 0.020] (length includes the shaft).
  color = can colour (#A9AFB7).
- gear: spur gear; its thinnest dimension is its axis. [0.030, 0.006, 0.030].
- generic: uses geometry and dimensions as described under "Geometry
  conventions" below.

Geometry conventions (generic parts only):
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
and rotation. Start poses should be plausible: a short distance above or
beside the final position (a few centimetres for small devices).
Use finite numbers and valid component references.

Each component must be introduced exactly once, in the step matching
its assembly_step. Step numbers must be consecutive, starting at 1.
Use a logical assembly sequence and meaningful final positions.

The camera must contain position and target arrays of three finite
numbers each. The viewer frames the whole device automatically, so the
camera mainly sets the viewing direction (position minus target).

Use arrays of strings for assumptions, validation_warnings, and
limitations. Include relevant uncertainties and safety warnings (for
example battery short circuits and polarity, LED current limiting, motor
stall current, mains voltage). Do not invent verified dimensions,
specifications, manufacturer part numbers, connections, or tests, and do not
claim the circuit has been simulated.

Create a simplified conceptual model for complex devices rather than
inventing large amounts of inaccurate detail. Do not add parts the device
does not need just to make it look detailed.

Placement rules (the model is rendered exactly as you place it):
- Y is up and the floor is y = 0. A part resting on the floor has position.y = height / 2.
- A part stacked on a parent has y = (parent.y + parent_height / 2) + part_height / 2.
  Example: a breadboard [0.165, 0.0085, 0.055] on the floor is at y = 0.00425. A resistor
  [0.030, 0.0025, 0.0025] lying on it is at y = 0.0085 + 0.00125 = 0.00975. An LED
  [0.0058, 0.030, 0.0058] whose leads sink 3 mm into the board is at
  y = 0.0085 - 0.003 + 0.015 = 0.0205.
- Compute positions from dimensions instead of estimating. Touching parts must touch or overlap; gaps larger than about 3% of the device size (and never more than 10 mm) are errors.
- Wires start inside one component's terminal and end inside the next (see the wire convention above).
- Circuits: mount everything on a breadboard or pcb, put the main parts in a row along X, and wire them in series (battery to resistor to LED and back).
- A generic cylinder lying along the X axis needs rotation [0, 0, 1.5708].
- Set attached_to on every part except the base.

Before returning the JSON, verify:
- All required fields exist, including component_type on every component.
- Component IDs are unique.
- component_type is one of the supported names and geometry names are supported.
- Dimensions are positive and contain exactly three numbers.
- Positions and rotations contain exactly three finite numbers.
- Colors and opacity values are valid.
- Assembly steps and component references are consistent.
- Animation types, durations, and destinations are valid.
- All output is valid JSON with no surrounding text.

Return the JSON object now.
"""


# Appended to every plan the viewer shows (once each).
VISUAL_APPROXIMATION_NOTICE = (
    "Detailed 3D component models are generic visual approximations. Resistor "
    "color bands, pin layouts, trace patterns and markings are illustrative and "
    "do not represent specific manufacturer parts."
)
CONCEPT_ONLY_NOTICE = (
    "This is a conceptual visualization only. It has not been electrically "
    "simulated, mechanically analyzed, or checked against datasheets, so a model "
    "that renders correctly is not evidence that the design works or is safe."
)
BATTERY_WARNING = (
    "Batteries can overheat, leak or catch fire if short-circuited, connected "
    "with reversed polarity or overloaded. Check polarity and add suitable "
    "current protection in a real build."
)
MOTOR_WARNING = (
    "Motors draw far more current when stalled than when running and can "
    "overheat or damage their supply or driver. Size the supply and protection "
    "for stall current."
)


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

        # component_type is new: plans without it (older plans) are drawn with
        # the original geometry renderer, which is what "generic" means.
        component_type = component.get("component_type")

        if component_type is None:
            component_type = "generic"
            component["component_type"] = component_type

        if (
            not isinstance(component_type, str)
            or component_type not in SUPPORTED_COMPONENT_TYPES
        ):
            raise ValueError(f"Unsupported component_type for {component_id}.")

        if component_type != "generic":
            # Detailed models ignore `geometry`; dimensions are their full
            # bounding box, so a box is the matching fallback shape.
            component["geometry"] = "box"

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

        # Never let an estimate be presented as a verified measurement.
        component["dimensions_verified"] = component.get("dimensions_verified") is True

    for component in components:
        attached_to = component.get("attached_to")
        if attached_to is not None and attached_to not in component_ids:
            raise ValueError(f"Unknown parent component {attached_to}.")

    if not all(isinstance(step, dict) for step in steps):
        raise ValueError("Every assembly step must be an object.")

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

        animations = step.get("animations", [])

        if not isinstance(animations, list):
            raise ValueError("animations must be a list.")

        for animation in animations:
            if not isinstance(animation, dict):
                raise ValueError("Every animation must be an object.")

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
    """Half-size of the component's axis-aligned bounding box.

    Detailed component types always use `dimensions` as a full bounding box
    (the validator sets their geometry to "box"), so they take the box branch.
    """
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


def check_layout(plan, tol=None):
    """Return a list of placement problems (floating or disconnected parts).

    By default the tolerance scales with the device (3% of its largest
    extent, between 2 mm and 10 mm) so breadboard-sized circuits are checked
    as strictly as furniture-sized ones.
    """
    boxes = {}

    for component in plan["components"]:
        half = _half_extents(component)
        boxes[component["id"]] = [
            (p - e, p + e) for p, e in zip(component["position"], half)
        ]

    if tol is None:
        extent = max(
            max(box[axis][1] for box in boxes.values())
            - min(box[axis][0] for box in boxes.values())
            for axis in range(3)
        )
        tol = min(0.01, max(0.002, 0.03 * extent))

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


def _append_unique(items, text):
    if text not in items:
        items.append(text)


def add_standard_notices(plan):
    """Add the fixed limitation / hazard notes that always apply to a plan."""
    types = {component["component_type"] for component in plan["components"]}

    if types - {"generic"}:
        _append_unique(plan["limitations"], VISUAL_APPROXIMATION_NOTICE)

    _append_unique(plan["limitations"], CONCEPT_ONLY_NOTICE)

    if "battery" in types:
        _append_unique(plan["validation_warnings"], BATTERY_WARNING)

    if "motor" in types:
        _append_unique(plan["validation_warnings"], MOTOR_WARNING)

    return plan


def generate_plan(question):
    """Ask the model for a plan; retry once if it is rejected or parts float."""
    base_input = (
        "Generate the device assembly plan for the following request. "
        "Return only a valid JSON object.\n\n"
        f"Device request: {question}"
    )

    prompt = base_input
    best_plan = None
    best_problems = []
    last_error = None
    started = time.monotonic()

    for attempt in range(2):
        if attempt == 1 and time.monotonic() - started > AI_TOTAL_BUDGET / 2:
            # The first answer was slow: don't make the user wait for a second one.
            if best_plan is None:
                raise last_error
            break

        try:
            ai_res = client.with_options(timeout=AI_REQUEST_TIMEOUT).responses.create(
                model="gpt-5.6-luna",
                instructions=SYSTEM_PROMPT,
                input=prompt,
                text={"format": {"type": "json_object"}},
            )
        except APITimeoutError:
            if best_plan is None:
                raise

            logger.warning("CircuitPilot retry timed out; keeping the first plan.")
            break

        if not ai_res.output_text:
            raise ValueError("The AI returned an empty response.")

        try:
            plan = validate_assembly_plan(json.loads(ai_res.output_text))
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc

            if attempt == 1 and best_plan is None:
                raise

            if attempt == 1:
                break  # keep the earlier valid plan and show its issues

            logger.warning("Retrying CircuitPilot plan after: %s", exc)
            prompt = (
                base_input
                + "\n\nYour previous JSON was rejected: "
                + str(exc)
                + "\nReturn the full corrected JSON object."
            )
            continue

        problems = check_layout(plan)

        if not problems:
            return add_standard_notices(plan)

        best_plan, best_problems = plan, problems
        prompt = (
            base_input
            + "\n\nYour previous plan had these placement problems. "
            "Return the full corrected JSON:\n- "
            + "\n- ".join(problems)
        )

    # Still imperfect after the retry: show the issues honestly.
    best_plan["validation_warnings"].extend(best_problems)
    return add_standard_notices(best_plan)


# ---------------------------------------------------------------------------
# Built-in sample plans for testing the 3D viewer without calling the AI.
# Only served when settings.DEBUG is True:  ?demo=components  or  ?demo=led
# ---------------------------------------------------------------------------

_BASE_TYPES = {"breadboard", "pcb", "microcontroller"}


def _component(cid, name, ctype, dims, position, color, function,
               attached_to=None, rotation=(0, 0, 0)):
    return {
        "id": cid,
        "name": name,
        "component_type": ctype,
        "geometry": "box",
        "dimensions": list(dims),
        "position": [round(v, 6) for v in position],
        "rotation": [round(v, 6) for v in rotation],
        "color": color,
        "material": "standard",
        "opacity": 1,
        "function": function,
        "dimensions_verified": False,
        "attached_to": attached_to,
    }


def _wire_between(start, end, overshoot=0.004, thickness=0.0018):
    """Dimensions, centre and rotation of a straight wire from start to end."""
    dx, dy, dz = (e - s for s, e in zip(start, end))
    horizontal = math.hypot(dx, dz)
    length = math.sqrt(dx * dx + dy * dy + dz * dz) + overshoot
    centre = [(s + e) / 2 for s, e in zip(start, end)]
    rotation = [0, -math.atan2(dz, dx), math.atan2(dy, horizontal)]
    return [length, thickness, thickness], centre, rotation


def _assemble_demo_plan(title, description, components, step_defs, camera,
                        assumptions, warnings):
    """Turn components + [(title, description, [ids])] into a full plan."""
    by_id = {c["id"]: c for c in components}
    steps = []

    for number, (step_title, step_description, ids) in enumerate(step_defs, 1):
        animations = []

        for cid in ids:
            component = by_id[cid]
            component["assembly_step"] = number
            to = component["position"]

            if component["component_type"] in _BASE_TYPES:
                kind, start = "fade_in", list(to)
            elif component["component_type"] == "wire":
                kind, start = "move_into_place", [to[0], to[1] + 0.03, to[2]]
            else:
                kind, start = "drop_into_place", [to[0], to[1] + 0.05, to[2]]

            animations.append({
                "component_id": cid,
                "type": kind,
                "from_position": start,
                "to_position": list(to),
                "from_rotation": list(component["rotation"]),
                "to_rotation": list(component["rotation"]),
                "duration_ms": 700,
            })

        steps.append({
            "step": number,
            "title": step_title,
            "description": step_description,
            "component_ids": list(ids),
            "animations": animations,
            "engineering_notes": [],
        })

    return {
        "title": title,
        "description": description,
        "units": "meters",
        "coordinate_system": "right_handed_y_up",
        "camera": camera,
        "components": components,
        "assembly_steps": steps,
        "assumptions": assumptions,
        "validation_warnings": warnings,
        "limitations": [],
    }


def build_component_showcase():
    """One of every detailed component type, laid out on the floor."""
    boards = [
        ("breadboard", "Breadboard (830 tie points)", "breadboard", [0.165, 0.0085, 0.055], "#ECEAE3"),
        ("pcb", "Bare circuit board", "pcb", [0.100, 0.0016, 0.070], "#0B6B3A"),
        ("dev_board", "Microcontroller development board", "microcontroller", [0.0686, 0.0135, 0.0534], "#1F6FB5"),
    ]
    electronics = [
        ("resistor", "470 \u03a9 resistor", "resistor", [0.030, 0.0025, 0.0025], "#D9B27C"),
        ("led", "Red LED (5 mm)", "led", [0.0058, 0.030, 0.0058], "#FF2A2A"),
        ("capacitor", "Electrolytic capacitor", "capacitor", [0.0065, 0.020, 0.0065], "#1D3F8F"),
        ("battery_aa", "AA cell", "battery", [0.0505, 0.0145, 0.0145], "#D6A21C"),
        ("battery_9v", "9 V battery", "battery", [0.0265, 0.0485, 0.0175], "#222831"),
        ("battery_coin", "Coin cell", "battery", [0.020, 0.0032, 0.020], "#C7CCD2"),
        ("wire", "Jumper wire", "wire", [0.080, 0.002, 0.002], "#D62B1F"),
    ]
    mechanical = [
        ("screw", "M3 x 10 screw", "screw", [0.0055, 0.012, 0.0055], "#B4B9BF"),
        ("motor", "Small DC motor", "motor", [0.020, 0.030, 0.020], "#A9AFB7"),
        ("gear", "Spur gear", "gear", [0.030, 0.006, 0.030], "#A7ADB5"),
    ]

    components = []

    def lay_out(rows, z, gap=0.012):
        total = sum(spec[3][0] for spec in rows) + gap * (len(rows) - 1)
        x = -total / 2
        for cid, name, ctype, dims, color in rows:
            components.append(_component(
                cid, name, ctype, dims,
                [x + dims[0] / 2, dims[1] / 2, z], color,
                f"Showcase of the {ctype} model",
            ))
            x += dims[0] + gap

    lay_out(boards, -0.05)
    lay_out(electronics + mechanical, 0.03)

    return _assemble_demo_plan(
        "Component showcase",
        "One of every detailed 3D component type, for testing the viewer.",
        components,
        [
            ("Boards", "Place the breadboard, bare PCB and development board.",
             [c[0] for c in boards]),
            ("Electronic parts", "Add the resistor, LED, capacitor, batteries and a wire.",
             [c[0] for c in electronics]),
            ("Mechanical parts", "Add the screw, motor and gear.",
             [c[0] for c in mechanical]),
        ],
        {"position": [0.3, 0.25, 0.4], "target": [0, 0.01, 0]},
        ["Sizes are typical real-world values used only for illustration."],
        ["Built-in sample plan for testing the viewer; it was not generated by the AI."],
    )


def build_led_demo():
    """LED circuit on a breadboard: 9 V battery -> resistor -> LED -> battery."""
    board_top = 0.0085
    battery_x = -0.12
    led_x = 0.036

    breadboard = _component(
        "breadboard", "Breadboard (830 tie points)", "breadboard",
        [0.165, 0.0085, 0.055], [0, 0.00425, 0], "#ECEAE3",
        "Holds and connects the circuit",
    )
    battery = _component(
        "battery_9v", "9 V battery", "battery",
        [0.0265, 0.0485, 0.0175], [battery_x, 0.02425, 0], "#222831",
        "Supplies power",
    )
    # Leads sink 3 mm into the board. Turned 90 degrees so anode and cathode
    # lie one behind the other (anode toward -Z).
    led = _component(
        "led", "Red LED (5 mm)", "led",
        [0.0058, 0.030, 0.0058], [led_x, board_top - 0.003 + 0.015, 0], "#FF2A2A",
        "Lights when current flows", "breadboard", [0, math.pi / 2, 0],
    )
    anode_z = -0.00128
    resistor = _component(
        "resistor", "470 \u03a9 resistor", "resistor",
        [0.030, 0.0025, 0.0025], [led_x - 0.015, board_top + 0.00125, anode_z],
        "#D9B27C", "Limits the LED current", "breadboard",
    )

    plus_terminal = [battery_x + 0.0265 * 0.24, 0.0475, 0]
    minus_terminal = [battery_x - 0.0265 * 0.24, 0.0475, 0]
    red_dims, red_pos, red_rot = _wire_between(
        plus_terminal, [led_x - 0.03, board_top + 0.0012, anode_z])
    black_dims, black_pos, black_rot = _wire_between(
        minus_terminal, [led_x, board_top + 0.001, -anode_z])
    wire_red = _component(
        "wire_red", "Red jumper wire", "wire", red_dims, red_pos, "#D62B1F",
        "Battery positive to resistor", "battery_9v", red_rot)
    wire_black = _component(
        "wire_black", "Black jumper wire", "wire", black_dims, black_pos, "#1A1A1A",
        "LED cathode back to battery negative", "battery_9v", black_rot)

    return _assemble_demo_plan(
        "LED circuit on a breadboard",
        "A 9 V battery, a current-limiting resistor and an LED on a breadboard.",
        [breadboard, battery, resistor, led, wire_red, wire_black],
        [
            ("Place the breadboard", "Set the breadboard on the workbench.", ["breadboard"]),
            ("Add the battery", "Position the 9 V battery beside the board.", ["battery_9v"]),
            ("Add the resistor", "Lay the resistor on the board.", ["resistor"]),
            ("Insert the LED", "Push the LED leads into the board, anode toward the resistor.", ["led"]),
            ("Connect the wires", "Wire battery positive to the resistor and the LED cathode to battery negative.",
             ["wire_red", "wire_black"]),
        ],
        {"position": [0.12, 0.1, 0.18], "target": [0, 0.01, 0]},
        [
            "The 470 \u03a9 resistor value is an illustrative assumption.",
            "Leads are drawn straight and are not bent into breadboard holes.",
        ],
        [
            "Built-in sample plan for testing the viewer; it was not generated by the AI.",
            "Check the resistor value against the LED's datasheet before building anything.",
        ],
    )


DEMO_PLAN_BUILDERS = {
    "components": build_component_showcase,
    "led": build_led_demo,
}


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

            except APITimeoutError:
                logger.warning("CircuitPilot request timed out.")
                error = (
                    "The AI took too long to respond. Please try again, "
                    "or describe a simpler device."
                )

            except Exception:
                logger.exception("CircuitPilot generation failed.")
                error = (
                    "Something went wrong while generating the device. "
                    "Please try again."
                )

    elif (
        getattr(settings, "DEBUG", False)
        and request.GET.get("demo") in DEMO_PLAN_BUILDERS
    ):
        try:
            assembly_plan = add_standard_notices(
                validate_assembly_plan(DEMO_PLAN_BUILDERS[request.GET["demo"]]())
            )
        except ValueError:
            logger.exception("Built-in demo plan is invalid.")
            error = "The built-in demo plan could not be loaded."

    return render(
        request,
        "cpai/main.html",
        {
            "question": question,
            "assembly_plan": assembly_plan,
            "error": error,
        },
    )