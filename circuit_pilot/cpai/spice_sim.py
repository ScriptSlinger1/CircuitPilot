"""Electrical model, SPICE netlist generation and a locked-down ngspice runner.

This module has no Django imports, so it can be tested on its own:

    python -m unittest test_spice_sim

Design rules
------------
* The 3D plan and the electrical model are separate. A component in the 3D plan may point at an electrical
  component through ``visual_id``, but connectivity ONLY comes from the explicit ``nets`` list. Two parts
  that look like they touch are never treated as connected.
* Nothing the AI (or a user) writes ever reaches the netlist as text. The netlist is assembled from validated
  numbers and from names generated here (n1, R1, D2 ...), and every line is checked against a whitelist
  before ngspice sees it. ngspice netlists can run shell commands and read files, so no free text is allowed.
* A result is only reported as "completed" when ngspice actually ran, exited cleanly, wrote a parseable
  raw file, and every expected vector was present. Otherwise the status says what went wrong.

Electrical circuit schema (``electrical`` key of a plan, schema_version 1)::

    {
      "components": [
        {"id": "e_r1", "visual_id": "resistor", "component_type": "resistor",
         "reference_designator": "R1",
         "parameters": {"resistance_ohms": 470},
         "parameter_basis": {"resistance_ohms": "assumed"},        # user_specified | assumed | estimated
         "pins": [{"name": "1"}, {"name": "2"}],                   # optional; must match the type
         "spice_model": {"preset": "led_red"}},                    # optional; presets only, never raw text
        ...],
      "nets": [{"name": "VCC", "connections": [{"ref": "B1", "pin": "positive"}, {"ref": "R1", "pin": "1"}]}, ...],
      "simulation_analysis": {"type": "op"}                        # or {"type": "tran", "step_s": 1e-4, "stop_s": 1e-2}
    }

``validate_electrical`` returns a normalised copy that also carries ``connections`` (flat list),
``simulation_status`` ("not_run"), ``simulation_results`` (None) and ``validation_warnings``.
"""

import math
import os
import re
import shutil
import signal
import struct
import subprocess
import tempfile

ELECTRICAL_SCHEMA_VERSION = 1

VT = 8.617333262e-5 * 300.15          # thermal voltage at ngspice's default 27 C

# --------------------------------------------------------------------------- supported parts

ELECTRICAL_TYPES = {
    "resistor": ("1", "2"),
    "capacitor": ("1", "2"),
    "inductor": ("1", "2"),
    "battery": ("positive", "negative"), # ideal DC voltage source
    "led": ("anode", "cathode"),
    "diode": ("anode", "cathode"),
    "zener_diode": ("anode", "cathode"),
    "switch": ("1", "2"), # static: closed = 1 mOhm, open = 1 GOhm
}

# Which 3D component types may represent each electrical type (a mismatch is reported as a warning).
VISUAL_TYPES_FOR = {
    "resistor": {"resistor"},
    "capacitor": {"capacitor", "ceramic_capacitor"},
    "inductor": set(),
    "battery": {"battery"},
    "led": {"led"},
    "diode": {"diode"},
    "zener_diode": {"zener_diode"},
    "switch": {"switch", "push_button"},
}

# Typical forward voltage at 10 mA used to build the generic LED / diode models. These are rough,
# colour-based ASSUMPTIONS, not datasheet values for any particular part.
LED_FORWARD_VOLTS = {"red": 1.95, "orange": 2.0, "yellow": 2.05, "green": 2.1, "blue": 3.0, "white": 3.0}
SILICON_DIODE_VF = 0.70
MODEL_TARGET_CURRENT = 0.010
LED_IDEALITY, LED_RS = 2.0, 2.0
SILICON_IDEALITY, SILICON_RS = 1.5, 0.5
LED_TYPICAL_MAX_A = 0.020 # typical continuous rating assumed for small indicator LEDs
RESISTOR_ASSUMED_RATING_W = 0.25

PARAMETER_BASES = {"user_specified", "assumed", "estimated"}

_PARAMS = {# type -> {name: (low, high, required, default)}
    "resistor": {"resistance_ohms": (1e-3, 1e12, True, None)},
    "capacitor": {"capacitance_f": (1e-15, 1.0, True, None), "initial_voltage_v": (-1000.0, 1000.0, False, 0.0)},
    "inductor": {"inductance_h": (1e-12, 1e3, True, None)},
    "battery": {"voltage_v": (-1000.0, 1000.0, True, None)},
    "zener_diode": {"zener_voltage_v": (1.0, 200.0, True, None)},
}

MAX_COMPONENTS = 200
MAX_NETS = 400
MAX_CONNECTIONS_PER_NET = 100
MAX_TRAN_POINTS = 20000
MAX_TRAN_STOP_S = 10.0

_REF = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,15}$")
_NET = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$|^0$")
_ID = re.compile(r"^[A-Za-z0-9_]{1,40}$")
_GROUND = {"0", "gnd"}


class ElectricalError(ValueError):
    """The electrical definition is invalid; ``problems`` lists everything that is wrong."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def is_ground(name):
    return isinstance(name, str) and name.lower() in _GROUND


# --------------------------------------------------------------------------- validation

def validate_electrical(data, visual_components=None):
    """Validate an ``electrical`` definition and return a normalised copy.

    ``visual_components`` is an optional {visual id: component_type} map of the 3D plan, used only to check
    that ``visual_id`` links make sense. Raises ElectricalError listing every problem found.
    """
    problems, warnings = [], []
    if not isinstance(data, dict):
        raise ElectricalError(["The electrical definition must be an object."])

    raw_components = data.get("components")
    raw_nets = data.get("nets")

    if not isinstance(raw_components, list) or not raw_components:
        raise ElectricalError(["The electrical definition needs a non-empty list of components."])

    if not isinstance(raw_nets, list) or not raw_nets:
        raise ElectricalError(["The electrical definition needs a non-empty list of nets (explicit connections)."])

    if len(raw_components) > MAX_COMPONENTS or len(raw_nets) > MAX_NETS:
        raise ElectricalError([f"Too large: at most {MAX_COMPONENTS} components and {MAX_NETS} nets are supported."])


    components, by_ref = [], {}
    for position, raw in enumerate(raw_components):
        label = f"component #{position + 1}"
        if not isinstance(raw, dict):
            problems.append(f"{label} must be an object.")
            continue

        ref = raw.get("reference_designator")
        if not isinstance(ref, str) or not _REF.match(ref):
            problems.append(f"{label} needs a reference_designator such as R1 (letters, digits, underscore).")
            continue

        label = ref
        if ref.lower() in by_ref:
            problems.append(f"Duplicate reference designator {ref}.")
            continue

        ctype = raw.get("component_type")
        if not isinstance(ctype, str) or ctype not in ELECTRICAL_TYPES:
            problems.append(f"{label}: unsupported electrical component_type {ctype!r} "
                            f"(supported: {', '.join(sorted(ELECTRICAL_TYPES))}).")
            continue

        pins = ELECTRICAL_TYPES[ctype]

        given_pins = raw.get("pins")
        if given_pins is not None:
            names = [p.get("name") for p in given_pins if isinstance(p, dict)] if isinstance(given_pins, list) else None

            if names is None or sorted(map(str, names)) != sorted(pins):
                problems.append(f"{label}: pins must be exactly {list(pins)} for a {ctype}.")

        raw_params = raw.get("parameters", {})
        if not isinstance(raw_params, dict):
            problems.append(f"{label}: parameters must be an object.")
            raw_params = {}

        params = {}
        for name, (low, high, required, default) in _PARAMS.get(ctype, {}).items():
            if name not in raw_params:
                if required:
                    problems.append(f"{label}: missing required parameter {name} (not guessed).")

                elif default is not None:
                    params[name] = default
                continue
            value = raw_params[name]

            if not _is_number(value) or not low <= value <= high:
                problems.append(f"{label}: {name} must be a number between {low:g} and {high:g}.")
            else:
                params[name] = float(value)

        if ctype == "battery" and params.get("voltage_v") == 0:
            warnings.append(f"{label}: a 0 V source supplies nothing.")

        if ctype == "led":
            colour = raw_params.get("color")
            if colour not in LED_FORWARD_VOLTS:
                problems.append(f"{label}: LED color must be one of {sorted(LED_FORWARD_VOLTS)} (not guessed).")

            else:
                params["color"] = colour

        if ctype == "switch":
            if raw_params.get("state") not in ("open", "closed"):
                problems.append(f"{label}: switch state must be 'open' or 'closed' (not guessed).")
            else:
                params["state"] = raw_params["state"]

        unknown = set(raw_params) - set(_PARAMS.get(ctype, {})) - {"color", "state"}
        if unknown:
            warnings.append(f"{label}: ignored unknown parameters {sorted(map(str, unknown))}.")


        basis = {}
        raw_basis = raw.get("parameter_basis", {})

        for name in params:
            declared = raw_basis.get(name) if isinstance(raw_basis, dict) else None
            basis[name] = declared if declared in PARAMETER_BASES else "assumed"

            if basis[name] != "user_specified":
                warnings.append(f"{label}: {name} is {basis[name]}, not specified by the user.")


        model = raw.get("spice_model")
        preset = None

        if model is not None:
            allowed = _model_presets_for(ctype)
            preset = model.get("preset") if isinstance(model, dict) else None

            if preset not in allowed:
                problems.append(f"{label}: spice_model must be {{'preset': one of {sorted(allowed)}}}; "
                                f"free-text models are not accepted.")
                preset = None

        visual_id = raw.get("visual_id")
        if visual_id is not None:
            if not isinstance(visual_id, str) or not _ID.match(visual_id):
                problems.append(f"{label}: invalid visual_id.")
                visual_id = None

            elif visual_components is not None:
                if visual_id not in visual_components:
                    warnings.append(f"{label}: visual_id {visual_id!r} is not a component of the 3D plan.")

                elif visual_components[visual_id] not in VISUAL_TYPES_FOR[ctype]:
                    warnings.append(f"{label}: the 3D part {visual_id!r} is a {visual_components[visual_id]}, "
                                    f"which does not look like a {ctype}.")

        entry = {
            "id": raw.get("id") if isinstance(raw.get("id"), str) and _ID.match(raw["id"]) else f"e_{ref.lower()}",
            "visual_id": visual_id,
            "component_type": ctype,
            "reference_designator": ref,
            "parameters": params,
            "parameter_basis": basis,
            "pins": [{"name": p, "net": None} for p in pins],
            "spice_model": {"preset": preset or _default_preset(ctype, params)},
        }
        components.append(entry)
        by_ref[ref.lower()] = entry

    # ---- nets: the only source of connectivity
    nets, seen_nets, used_pins, ground_nets = [], set(), {}, []
    for position, raw in enumerate(raw_nets):
        if not isinstance(raw, dict):
            problems.append(f"net #{position + 1} must be an object.")
            continue

        name = raw.get("name")
        if not isinstance(name, str) or not _NET.match(name):
            problems.append(f"net #{position + 1} needs a name of letters, digits and underscores (starting with a letter).")
            continue

        if name.lower() in seen_nets:
            problems.append(f"Duplicate net name {name}.")
            continue

        seen_nets.add(name.lower())
        if is_ground(name):
            ground_nets.append(name)

        connections = raw.get("connections")
        if not isinstance(connections, list) or not connections or len(connections) > MAX_CONNECTIONS_PER_NET:
            problems.append(f"net {name} needs 1 to {MAX_CONNECTIONS_PER_NET} connections.")
            continue

        clean = []

        for c in connections:
            ref = c.get("ref") if isinstance(c, dict) else None
            pin = str(c.get("pin")) if isinstance(c, dict) and c.get("pin") is not None else None
            target = by_ref.get(ref.lower()) if isinstance(ref, str) else None

            if target is None:
                problems.append(f"net {name} refers to unknown component {ref!r}.")
                continue

            if pin not in ELECTRICAL_TYPES[target["component_type"]]:
                problems.append(f"net {name}: {target['reference_designator']} has no pin {pin!r}.")
                continue

            key = (target["reference_designator"], pin)
            if key in used_pins:
                problems.append(f"{key[0]}.{pin} is connected to two nets ({used_pins[key]} and {name}).")
                continue

            used_pins[key] = name
            next(p for p in target["pins"] if p["name"] == pin)["net"] = name
            clean.append({"ref": target["reference_designator"], "pin": pin})

        if len(clean) == 1:
            warnings.append(f"net {name} has only one connection.")

        nets.append({"name": name, "connections": clean})


    for entry in components:
        for pin in entry["pins"]:
            if pin["net"] is None:
                problems.append(f"{entry['reference_designator']}.{pin['name']} is not connected to any net "
                                f"(connections are never inferred from the 3D layout).")

    if len(ground_nets) != 1:
        problems.append("Exactly one ground net named GND (or 0) is required as the voltage reference "
                        f"(found {len(ground_nets)}).")
    if not any(c["component_type"] == "battery" for c in components):
        warnings.append("The circuit has no voltage source, so every node will sit at 0 V.")

    analysis = _validate_analysis(data.get("simulation_analysis", {"type": "op"}), problems)

    if problems:
        raise ElectricalError(problems)

    return {
        "schema_version": ELECTRICAL_SCHEMA_VERSION,
        "components": components,
        "nets": nets,
        "connections": [{"net": n["name"], **c} for n in nets for c in n["connections"]],
        "simulation_analysis": analysis,
        "simulation_status": "not_run",
        "simulation_results": None,
        "validation_warnings": warnings,
    }


def _model_presets_for(ctype):
    if ctype == "led":
        return {f"led_{colour}" for colour in LED_FORWARD_VOLTS}

    if ctype in ("diode", "zener_diode"):
        return {"silicon_generic"}

    return set()


def _default_preset(ctype, params):
    if ctype == "led":
        return f"led_{params.get('color', 'red')}"

    if ctype in ("diode", "zener_diode"):
        return "silicon_generic"

    return None


def _validate_analysis(raw, problems):
    if not isinstance(raw, dict) or raw.get("type") not in ("op", "tran"):
        problems.append("simulation_analysis.type must be 'op' or 'tran' (AC analysis is not implemented yet).")
        return {"type": "op"}

    if raw["type"] == "op":
        return {"type": "op"}

    step, stop = raw.get("step_s"), raw.get("stop_s")

    if not (_is_number(step) and _is_number(stop) and 1e-12 <= step <= stop <= MAX_TRAN_STOP_S):
        problems.append(f"tran analysis needs numbers with 1e-12 <= step_s <= stop_s <= {MAX_TRAN_STOP_S:g}.")
        return {"type": "op"}

    if stop / step > MAX_TRAN_POINTS:
        problems.append(f"tran analysis would need more than {MAX_TRAN_POINTS} points.")
        return {"type": "op"}

    return {"type": "tran", "step_s": float(step), "stop_s": float(stop),
            "use_initial_conditions": raw.get("use_initial_conditions") is True}


# --------------------------------------------------------------------------- models and netlist

def _num(value):
    return format(float(value), ".9g")


def _diode_params(vf, ideality, rs):
    """Saturation current that makes the forward drop equal `vf` at MODEL_TARGET_CURRENT."""
    saturation = MODEL_TARGET_CURRENT * math.exp(-(vf - MODEL_TARGET_CURRENT * rs) / (ideality * VT))
    return saturation, ideality, rs


def model_parameters(entry):
    """(IS, N, RS) of the generic diode model used for an electrical LED / diode / zener entry."""
    preset = entry["spice_model"]["preset"]

    if preset.startswith("led_"):
        return _diode_params(LED_FORWARD_VOLTS[preset[4:]], LED_IDEALITY, LED_RS)

    return _diode_params(SILICON_DIODE_VF, SILICON_IDEALITY, SILICON_RS)


def build_netlist(electrical):
    """Build the SPICE netlist from a NORMALISED electrical definition.

    Returns (netlist_text, meta). Two-terminal parts get a 0 V "ammeter" source in series so each branch current
    can be read back from the raw file without relying on simulator-specific syntax.
    """
    node = {}
    counter = 0
    for net in sorted(electrical["nets"], key=lambda n: n["name"].lower()):
        if is_ground(net["name"]):
            node[net["name"]] = "0"

        else:
            counter += 1
            node[net["name"]] = f"n{counter}"

    lines = ["* CircuitPilot generated netlist"]
    cards, probes, sources = {}, [], []
    index = {"R": 0, "C": 0, "L": 0, "D": 0, "VB": 0, "VP": 0}
    uic = False
    analysis = electrical["simulation_analysis"]

    for entry in electrical["components"]:
        pin_node = {p["name"]: node[p["net"]] for p in entry["pins"]}
        ctype, params, ref = entry["component_type"], entry["parameters"], entry["reference_designator"]
        first, second = ELECTRICAL_TYPES[ctype]

        if ctype == "battery":
            index["VB"] += 1
            name = f"VB{index['VB']}"
            lines.append(f"{name} {pin_node[first]} {pin_node[second]} DC {_num(params['voltage_v'])}")
            sources.append({"ref": ref, "type": ctype, "element": name.lower(), "n1": pin_node[first], "n2": pin_node[second]})
            continue

        index["VP"] += 1
        probe, inner = f"VP{index['VP']}", f"p{index['VP']}"
        lines.append(f"{probe} {pin_node[first]} {inner} DC 0")
        probes.append({"ref": ref, "type": ctype, "element": probe.lower(), "n1": pin_node[first], "n2": pin_node[second]})
        a, b = inner, pin_node[second]

        if ctype in ("resistor", "switch"):
            index["R"] += 1
            value = params["resistance_ohms"] if ctype == "resistor" else (1e-3 if params["state"] == "closed" else 1e9)
            lines.append(f"R{index['R']} {a} {b} {_num(value)}")

        elif ctype == "capacitor":
            index["C"] += 1
            tail = ""

            if analysis["type"] == "tran" and analysis.get("use_initial_conditions"):
                tail = f" IC={_num(params.get('initial_voltage_v', 0.0))}"
                uic = True

            lines.append(f"C{index['C']} {a} {b} {_num(params['capacitance_f'])}{tail}")

        elif ctype == "inductor":
            index["L"] += 1
            lines.append(f"L{index['L']} {a} {b} {_num(params['inductance_h'])}")

        else: # led, diode, zener_diode
            index["D"] += 1
            model = f"DM{index['D']}"
            saturation, ideality, rs = model_parameters(entry)
            card = f"IS={_num(saturation)} N={_num(ideality)} RS={_num(rs)}"

            if ctype == "zener_diode":
                card += f" BV={_num(params['zener_voltage_v'])} IBV=0.005"

            cards[model] = f".model {model} D({card})"
            lines.append(f"D{index['D']} {a} {b} {model}")

    lines.extend(cards[name] for name in sorted(cards))

    if analysis["type"] == "op":
        lines.append(".op")

    else:
        lines.append(f".tran {_num(analysis['step_s'])} {_num(analysis['stop_s'])}" + (" uic" if uic else ""))

    lines.append(".end")
    text = "\n".join(lines) + "\n"
    assert_safe_netlist(text)
    meta = {"nodes": {v: k for k, v in node.items()}, "probes": probes, "sources": sources, "ground": "0"}

    return text, meta


_NUM = r"[+-]?\d{1,12}(?:\.\d{1,12})?(?:e[+-]?\d{1,2})?"
_ALLOWED_LINES = [
    re.compile(r"^\* [A-Za-z0-9 _.,:()=+/-]*$"),
    re.compile(rf"^V[BP][0-9]+ [A-Za-z0-9_]+ [A-Za-z0-9_]+ DC {_NUM}$"),
    re.compile(rf"^[RCL][0-9]+ [A-Za-z0-9_]+ [A-Za-z0-9_]+ {_NUM}(?: IC={_NUM})?$"),
    re.compile(r"^D[0-9]+ [A-Za-z0-9_]+ [A-Za-z0-9_]+ DM[0-9]+$"),
    re.compile(rf"^\.model DM[0-9]+ D\((?:[A-Z]+={_NUM})(?: [A-Z]+={_NUM})*\)$"),
    re.compile(r"^\.op$"),
    re.compile(rf"^\.tran {_NUM} {_NUM}(?: uic)?$"),
    re.compile(r"^\.end$"),
]

MAX_NETLIST_BYTES = 200_000


class NetlistSafetyError(ValueError):
    pass


def assert_safe_netlist(text):
    """Refuse any netlist line that is not one of the exact shapes this module generates."""
    if not isinstance(text, str) or len(text.encode("utf-8", "replace")) > MAX_NETLIST_BYTES:
        raise NetlistSafetyError("netlist is too large")

    for number, line in enumerate(text.split("\n")[:-1] if text.endswith("\n") else text.split("\n"), 1):
        if len(line) > 300 or not any(pattern.match(line) for pattern in _ALLOWED_LINES):
            raise NetlistSafetyError(f"netlist line {number} is not an allowed statement")


# --------------------------------------------------------------------------- hand calculation

def analytic_estimate(electrical):
    """Hand calculation for a single series loop: battery -> resistor -> LED/diode -> back to the battery.

    Returns None for any other topology. `fixed_vf_current_a` is the textbook (Vs - Vf) / R estimate;
    `model_current_a` solves the same diode equation the netlist uses, so it should match SPICE closely.
    """
    comps = {c["reference_designator"]: c for c in electrical["components"]}
    kinds = sorted(c["component_type"] for c in comps.values())

    if kinds not in (["battery", "led", "resistor"], ["battery", "diode", "resistor"]):
        return None


    pin_net = {(c["reference_designator"], p["name"]): p["net"] for c in comps.values() for p in c["pins"]}
    members = {n["name"]: {(x["ref"], x["pin"]) for x in n["connections"]} for n in electrical["nets"]}

    if any(len(m) != 2 for m in members.values()):
        return None


    ref = {kind: next(r for r, c in comps.items() if c["component_type"] == kind) for kind in kinds}
    diode_kind = "led" if "led" in kinds else "diode"
    d_ref, r_ref, b_ref = ref[diode_kind], ref["resistor"], ref["battery"]

    if pin_net[(b_ref, "positive")] != pin_net[(r_ref, "1")] or pin_net[(r_ref, "2")] != pin_net[(d_ref, "anode")] \
            or pin_net[(d_ref, "cathode")] != pin_net[(b_ref, "negative")]:
            return None


    supply = comps[b_ref]["parameters"]["voltage_v"]
    resistance = comps[r_ref]["parameters"]["resistance_ohms"]
    saturation, ideality, rs = model_parameters(comps[d_ref])

    vf = LED_FORWARD_VOLTS[comps[d_ref]["parameters"]["color"]] if diode_kind == "led" else SILICON_DIODE_VF
    drop = lambda i: ideality * VT * math.log(i / saturation + 1.0) + i * rs
    low, high = 0.0, max(supply, 0.0) / resistance

    for _ in range(200): # drop(i) + i*R is increasing in i: bisection is safe
        mid = (low + high) / 2

        if drop(mid) + mid * resistance > supply:
            high = mid

        else:
            low = mid

    current = (low + high) / 2 if supply > 0 else 0.0

    return {
        "topology": f"battery-resistor-{diode_kind}",
        "supply_v": supply, "resistance_ohms": resistance,
        "fixed_vf_current_a": max(0.0, (supply - vf) / resistance),
        "model_current_a": current,
        "model_diode_voltage_v": drop(current) if current > 0 else 0.0,
        "diode_ref": d_ref,
    }


# --------------------------------------------------------------------------- raw file

class RawFileError(ValueError):
    pass


def parse_rawfile(data):
    """Parse the first plot of a SPICE3/ngspice raw file (binary or ASCII).

    Returns {"plotname", "complex", "variables": [{"index", "name", "type"}], "rows": [[float, ...], ...]}.
    """
    for marker in (b"Binary:\n", b"Values:\n"):
        at = data.find(marker)

        if at >= 0:
            break
    else:

        raise RawFileError("raw file has no Binary:/Values: section")


    header = data[:at].decode("latin-1").split("\n")
    body = data[at + len(marker):]
    binary = marker.startswith(b"Binary")

    fields, variables, reading_vars = {}, [], False
    for line in header:
        if reading_vars:
            parts = line.split()

            if len(parts) >= 3 and parts[0].isdigit():
                variables.append({"index": int(parts[0]), "name": parts[1], "type": parts[2].lower()})
                continue

            reading_vars = False

        if ":" in line:
            key, _, value = line.partition(":")
            fields[key.strip().lower()] = value.strip()
            reading_vars = key.strip().lower() == "variables"

    try:
        n_vars, n_points = int(fields["no. variables"]), int(fields["no. points"])

    except (KeyError, ValueError):
        raise RawFileError("raw file header is missing the variable or point count")

    if len(variables) != n_vars or n_vars < 1 or n_points < 0 or n_vars * n_points > 5_000_000:
        raise RawFileError("raw file header is inconsistent")


    is_complex = "complex" in fields.get("flags", "").lower()

    rows = []
    if binary:
        per_value = 16 if is_complex else 8
        need = n_vars * n_points * per_value

        if len(body) < need:
            raise RawFileError(f"raw file is truncated ({len(body)} of {need} data bytes)")

        values = struct.unpack("<" + "d" * (n_vars * n_points * (2 if is_complex else 1)), body[:need])
        step = n_vars * (2 if is_complex else 1)

        for p in range(n_points):
            row = values[p * step:(p + 1) * step]
            rows.append([complex(row[2 * k], row[2 * k + 1]) for k in range(n_vars)] if is_complex else list(row))

    else:
        tokens, pos = body.decode("latin-1").split(), 0
        for _ in range(n_points):
            pos += 1 # point index
            row = []

            for _k in range(n_vars):
                token = tokens[pos]
                pos += 1

                if is_complex:
                    re_part, _, im_part = token.partition(",")
                    row.append(complex(float(re_part), float(im_part)))

                else:
                    row.append(float(token))

            rows.append(row)

    return {"plotname": fields.get("plotname", ""), "complex": is_complex, "variables": variables, "rows": rows}


def _normalise_variable(variable):
    """('voltage'|'current'|other, lower-case node or source name), tolerant of v(x), i(x) and x#branch styles."""
    name = variable["name"].lower()
    match = re.fullmatch(r"[vi]\((.+)\)", name)

    if match:
        name = match.group(1)

    if name.endswith("#branch"):
        name = name[: -len("#branch")]

    return variable["type"], name


# --------------------------------------------------------------------------- running ngspice

class NgspiceRunError(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message
        super().__init__(message)


MAX_RAW_BYTES = 20_000_000
MAX_LOG_BYTES = 1_000_000
_NO_CONVERGENCE = ("no convergence", "singular matrix", "timestep too small", "iteration limit", "too many iterations")


def find_ngspice(configured=None):
    return configured if configured and os.path.isfile(configured) else shutil.which("ngspice")


def _apply_limits(cpu_seconds):
    def apply():
        import resource                                # POSIX only
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_RAW_BYTES + MAX_LOG_BYTES, MAX_RAW_BYTES + MAX_LOG_BYTES))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))

    return apply


def run_ngspice(netlist, executable, timeout=10.0):
    """Run ngspice in batch mode on a generated netlist. Returns (raw_bytes, log_text).

    * the command line is a fixed argument list (no shell);
    * `-n` skips any .spiceinit, the environment is scrubbed, and the working directory is a private temp dir;
    * CPU, memory and file-size limits apply on POSIX, and a wall-clock timeout kills the whole process group;
    * the netlist is re-checked against the statement whitelist immediately before it is written.
    """
    assert_safe_netlist(netlist)
    posix = os.name == "posix"

    with tempfile.TemporaryDirectory(prefix="cpai_spice_") as tmp:
        circuit, raw_path, log_path = (os.path.join(tmp, n) for n in ("circuit.cir", "out.raw", "out.log"))

        with open(circuit, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(netlist)
        env = {"PATH": os.defpath, "HOME": tmp, "TMPDIR": tmp, "LC_ALL": "C", "SPICE_ASCIIRAWFILE": "0"}

        if os.name == "nt":
            env["SystemRoot"] = os.environ.get("SystemRoot", r"C:\Windows")

        try:
            process = subprocess.Popen(
                [executable, "-b", "-n", "-r", raw_path, "-o", log_path, circuit],
                cwd=tmp, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                shell=False, start_new_session=posix,
                preexec_fn=_apply_limits(int(math.ceil(timeout)) + 2) if posix else None)

        except OSError as error:
            raise NgspiceRunError("unavailable", f"could not start ngspice: {error.strerror or error}")

        try:
            stdout, stderr = process.communicate(timeout=timeout)

        except subprocess.TimeoutExpired:
            try:
                if posix:
                    os.killpg(process.pid, signal.SIGKILL)

                else:
                    process.kill()

            finally:
                process.communicate()

            raise NgspiceRunError("failed", f"ngspice did not finish within {timeout:g} s and was stopped")

        log = ""
        if os.path.isfile(log_path):
            with open(log_path, "rb") as handle:
                log = handle.read(MAX_LOG_BYTES).decode("utf-8", "replace")

        log += "\n" + (stdout or b"").decode("utf-8", "replace")[:MAX_LOG_BYTES] + "\n" + (stderr or b"").decode("utf-8", "replace")[:MAX_LOG_BYTES]
        lowered = log.lower()


        if any(token in lowered for token in _NO_CONVERGENCE):
            raise NgspiceRunError("not_converged", "ngspice reported a convergence problem for this circuit")

        if process.returncode != 0:
            raise NgspiceRunError("failed", f"ngspice exited with status {process.returncode}: {_first_error(log)}")

        if not os.path.isfile(raw_path):
            raise NgspiceRunError("failed", f"ngspice produced no results file: {_first_error(log)}")

        if os.path.getsize(raw_path) > MAX_RAW_BYTES:
            raise NgspiceRunError("failed", "the results file was too large")

        with open(raw_path, "rb") as handle:
            return handle.read(), log


def _first_error(log):
    for line in log.splitlines():
        if "error" in line.lower():
            return re.sub(r"[^\x20-\x7e]", "?", line.strip())[:200]
    return "no error text was reported"


# --------------------------------------------------------------------------- results

def _finite(value):
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def interpret_results(electrical, meta, parsed, analysis):
    """Turn a parsed raw file into the structured result. Raises NgspiceRunError if expected vectors are missing."""
    if not parsed["rows"]:
        raise NgspiceRunError("failed", "ngspice returned no data points")

    if parsed["complex"]:
        raise NgspiceRunError("failed", "unexpected complex results for a DC/transient analysis")

    voltages, currents, trace_names = {}, {}, {}
    for variable in parsed["variables"]:
        kind, name = _normalise_variable(variable)
        column = variable["index"]

        if kind == "voltage":
            voltages[name] = column

        elif kind == "current":
            currents[name] = column

        elif kind == "time":
            trace_names["time"] = column

    last = parsed["rows"][-1]

    def node_voltage(spice_node):
        if spice_node == meta["ground"]:
            return 0.0

        if spice_node not in voltages:
            raise NgspiceRunError("failed", f"ngspice did not report node {spice_node}")

        return last[voltages[spice_node]]

    def branch_current(element):
        if element not in currents:
            raise NgspiceRunError("failed", f"ngspice did not report the current of {element}")

        return last[currents[element]]

    nodes = [{"net": net, "voltage_v": _finite(node_voltage(spice))}
             for spice, net in sorted(meta["nodes"].items(), key=lambda kv: kv[1].lower())]


    components, absorbed = [], 0.0

    for probe in meta["probes"]:
        current = branch_current(probe["element"])
        voltage = node_voltage(probe["n1"]) - node_voltage(probe["n2"])
        absorbed += voltage * current
        components.append({"ref": probe["ref"], "type": probe["type"], "voltage_v": _finite(voltage),
                           "current_a": _finite(current), "power_w": _finite(voltage * current)})

    sources, delivered = [], 0.0

    for source in meta["sources"]:
        supplied = -branch_current(source["element"]) # SPICE: positive branch current flows INTO the + terminal
        voltage = node_voltage(source["n1"]) - node_voltage(source["n2"])
        delivered += voltage * supplied
        sources.append({"ref": source["ref"], "type": source["type"], "voltage_v": _finite(voltage),
                        "current_a": _finite(supplied), "power_w": _finite(voltage * supplied)})

    warnings = []

    if analysis["type"] == "op":
        scale = max(abs(delivered), abs(absorbed), 1e-9)

        if abs(delivered - absorbed) > 1e-3 * scale + 1e-9:
            warnings.append("Power delivered by the sources does not match power absorbed by the parts; "
                            "the result may be inconsistent.")

    for item in components:
        kind, current, power = item["type"], item["current_a"], item["power_w"]

        if kind == "led" and current is not None:

            if current > LED_TYPICAL_MAX_A:
                warnings.append(f"{item['ref']} carries {current * 1000:.1f} mA, above the generic "
                                f"{LED_TYPICAL_MAX_A * 1000:.0f} mA indicator-LED limit assumed here.")

            elif current < 1e-4:
                warnings.append(f"{item['ref']} carries almost no current, so the LED would be dark.")

        if kind == "resistor" and power is not None and power > RESISTOR_ASSUMED_RATING_W:
            warnings.append(f"{item['ref']} dissipates {power:.3f} W, above the {RESISTOR_ASSUMED_RATING_W} W rating assumed here.")

    result = {"nodes": nodes, "components": components, "sources": sources, "warnings": warnings}
    if analysis["type"] == "tran" and "time" in trace_names:
        stride = max(1, len(parsed["rows"]) // 200)
        rows = parsed["rows"][::stride]
        result["traces"] = {
            "time_s": [_finite(r[trace_names["time"]]) for r in rows],
            "nets": {meta["nodes"][s]: [_finite(r[voltages[s]]) for r in rows] for s in meta["nodes"] if s in voltages},
        }

    return result


MODEL_NOTES = [
    "LEDs and diodes use generic SPICE models whose forward voltage is a typical-by-colour assumption, not a datasheet value.",
    "Batteries are ideal voltage sources: real cells have internal resistance and their voltage changes with load and charge.",
    "Ratings used for warnings (20 mA LED, 0.25 W resistor) are generic assumptions, not limits of any specific part.",
]


def simulate(electrical, visual_components=None, ngspice_path=None, timeout=10.0):
    """Validate, build, run and interpret. Always returns a JSON-serialisable dict with a `status`:

    completed | invalid_circuit | unavailable | not_converged | failed
    Only "completed" means a real ngspice run finished and produced every expected value.
    """
    try:
        normalised = validate_electrical(electrical, visual_components)

    except ElectricalError as error:
        return {"status": "invalid_circuit", "message": "; ".join(error.problems), "problems": error.problems}


    analysis = normalised["simulation_analysis"]
    netlist, meta = build_netlist(normalised)
    executable = find_ngspice(ngspice_path)

    if not executable:
        return {"status": "unavailable", "analysis": analysis,
                "message": "ngspice is not installed on the server (install it, or set CIRCUITPILOT_NGSPICE_PATH). "
                           "Nothing was simulated, so the circuit is unverified."}

    try:
        raw, log = run_ngspice(netlist, executable, timeout)
        parsed = parse_rawfile(raw)
        body = interpret_results(normalised, meta, parsed, analysis)

    except NgspiceRunError as error:
        return {"status": error.status, "message": error.message, "analysis": analysis}

    except RawFileError as error:
        return {"status": "failed", "message": f"could not read ngspice's results: {error}", "analysis": analysis}

    version = re.search(r"ngspice[- ](\d+(?:\.\d+)*)", log)
    estimate = analytic_estimate(normalised)

    if estimate:
        diode = next((c for c in body["components"] if c["ref"] == estimate["diode_ref"]), None)

        if diode and diode["current_a"] is not None and analysis["type"] == "op":
            reference = estimate["model_current_a"]
            estimate["simulated_current_a"] = diode["current_a"]
            estimate["agrees"] = abs(diode["current_a"] - reference) <= 0.02 * max(reference, 1e-9) + 1e-9

            if not estimate["agrees"]:
                body["warnings"].append("The simulated current differs from the hand calculation by more than 2%.")

    body.update({
        "status": "completed", "converged": True, "analysis": analysis,
        "engine": {"name": "ngspice", "version": f"ngspice-{version.group(1)}" if version else None},
        "analytic": estimate,
        "notes": normalised["validation_warnings"] + MODEL_NOTES,
    })

    return body


# --------------------------------------------------------------------------- ready-made circuit

def build_led_circuit(supply_v=9.0, resistance_ohms=470.0, color="red", visual_ids=None):
    """Electrical definition of a battery -> resistor -> LED loop (the first test circuit)."""
    visual_ids = visual_ids or {}
    assumed = "assumed"

    return {
        "components": [
            {"id": "e_b1", "visual_id": visual_ids.get("battery"), "component_type": "battery", "reference_designator": "B1",
             "parameters": {"voltage_v": supply_v}, "parameter_basis": {"voltage_v": assumed}},
            {"id": "e_r1", "visual_id": visual_ids.get("resistor"), "component_type": "resistor", "reference_designator": "R1",
             "parameters": {"resistance_ohms": resistance_ohms}, "parameter_basis": {"resistance_ohms": assumed}},
            {"id": "e_d1", "visual_id": visual_ids.get("led"), "component_type": "led", "reference_designator": "D1",
             "parameters": {"color": color}, "parameter_basis": {"color": assumed}},
        ],
        "nets": [
            {"name": "VCC", "connections": [{"ref": "B1", "pin": "positive"}, {"ref": "R1", "pin": "1"}]},
            {"name": "N_LED", "connections": [{"ref": "R1", "pin": "2"}, {"ref": "D1", "pin": "anode"}]},
            {"name": "GND", "connections": [{"ref": "D1", "pin": "cathode"}, {"ref": "B1", "pin": "negative"}]},
        ],
        "simulation_analysis": {"type": "op"},
    }
