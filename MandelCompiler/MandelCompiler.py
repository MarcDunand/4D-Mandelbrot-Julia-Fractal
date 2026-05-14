"""
Modern WHCS compiler.

This reads a lightweight command file and renders a WHCS animation through
WHCS_PySide6.generateFromCode(). The output animation length is determined by
compiler commands, not by Tres. Tres remains the resolution/scrubber concept for
WHCS's T dimension, while Tcur is the explicit T slice rendered at each output
frame.

Command file rules:
    - Blank lines are ignored.
    - Anything after # is a comment.
    - Two-token lines set initial values:
          VT VALUE
        - Four-token lines animate or cut values:
          VT FI FF VF
      Five-token lines do the same thing, but specify interpolation mode:
          VT FI FF VF MODE

      MODE is optional. Valid modes:
          linear  - normal linear interpolation. This is the default.
          smooth  - smoothstep easing: u*u*(3 - 2*u)

      where VT is the value type, FI is the first output frame, FF is the first
      frame after the interpolation, and VF is the final value.
    - Numeric commands interpolate over [FI, FF). If FI == FF, the command is a
      hard cut at FI.
    - indicator changes are always hard cuts at FI.
    - Parameter names are case-insensitive.
    - Rotations are user-facing degrees in the command file and radians inside
      WHCS_PySide6.
"""

from __future__ import annotations

import argparse
import copy
import os
import shlex
import sys
from pathlib import Path
from dataclasses import fields
from typing import Dict, List, NamedTuple, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import WHCS_PySide6 as WHCS


ROTATION_FIELDS = {"XYrot", "XTrot", "XHrot", "YTrot", "YHrot", "HTrot"}
INTEGER_FIELDS = {"Xres", "Yres", "Tres", "Cres", "precision"}
INIT_ONLY_FIELDS = {"Xres", "Yres", "Tres"}
NONINTERPOLATED_FIELDS = {"indicator"}
INTERPOLATION_MODES = {"linear", "smooth"}

# These are the values that should be deliberately set at the top of the command
# file. They match the modern WHCS_PySide6 ViewerState, plus the explicit Tcur
# slice used by compiled renders.
REQUIRED_INITIAL_FIELDS = {
    "indicator",
    "Xmin", "Xmax", "Xres",
    "Ymin", "Ymax", "Yres",
    "Tmin", "Tmax", "Tres", "Tcur",
    "Cmin", "Cmax", "Cres",
    "XYrot", "XTrot", "XHrot", "YTrot", "YHrot", "HTrot",
    "precision",
}


class CompilerCommand(NamedTuple):
    field: str
    fi: int
    ff: int
    value: object
    mode: str
    line_no: int
    source: str


def smoothstep(alpha: float) -> float:
    """Classic smoothstep easing: zero velocity at both ends."""
    return alpha * alpha * (3.0 - 2.0 * alpha)


def apply_interpolation_mode(alpha: float, mode: str) -> float:
    if mode == "linear":
        return alpha
    if mode == "smooth":
        return smoothstep(alpha)
    raise ValueError(f"Unknown interpolation mode: {mode}")


def canonical_field(name: str) -> str:
    """Map case-insensitive, user-facing command names onto ViewerState fields."""
    key = name.strip().lower()
    aliases = {
        "indicator": "indicator",
        "xmin": "Xmin", "xmax": "Xmax", "xres": "Xres",
        "ymin": "Ymin", "ymax": "Ymax", "yres": "Yres",
        "tmin": "Tmin", "tmax": "Tmax", "tres": "Tres", "tcur": "Tcur",
        "cmin": "Cmin", "cmax": "Cmax", "cres": "Cres",
        "precision": "precision", "prec": "precision",

        # User-facing color-axis rotation names. Internally, the old engine name
        # for the color axis is still H, so XC/YC/CT map onto XH/YH/HT.
        "xyrot": "XYrot", "xy": "XYrot",
        "xtrot": "XTrot", "xt": "XTrot",
        "xcrot": "XHrot", "xc": "XHrot", "xhrot": "XHrot", "xh": "XHrot",
        "ytrot": "YTrot", "yt": "YTrot",
        "ycrot": "YHrot", "yc": "YHrot", "yhrot": "YHrot", "yh": "YHrot",
        "ctrot": "HTrot", "ct": "HTrot", "tcrot": "HTrot", "tc": "HTrot",
        "htrot": "HTrot", "ht": "HTrot",
    }
    if key not in aliases:
        valid = ", ".join(sorted(aliases.keys()))
        raise ValueError(f"Unknown value type '{name}'. Valid names include: {valid}")
    return aliases[key]


def convert_value(field: str, raw_value: str):
    """Convert command-file strings into internal ViewerState values."""
    if field == "indicator":
        if raw_value not in WHCS.INDICATORS:
            valid = ", ".join(WHCS.INDICATORS.keys())
            raise ValueError(f"Unknown indicator '{raw_value}'. Valid indicators: {valid}")
        return raw_value

    if field in ROTATION_FIELDS:
        return float(raw_value) * np.pi / 180.0

    if field in INTEGER_FIELDS:
        return int(round(float(raw_value)))

    return float(raw_value)


def strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def parse_command_file(path: str) -> Tuple[WHCS.ViewerState, List[CompilerCommand]]:
    init_values: Dict[str, object] = {}
    commands: List[CompilerCommand] = []

    with open(path, "r", encoding="utf-8") as file:
        for line_no, raw_line in enumerate(file, start=1):
            cleaned = strip_comment(raw_line)
            if not cleaned:
                continue

            parts = shlex.split(cleaned)
            if len(parts) == 2:
                field = canonical_field(parts[0])
                if field in init_values:
                    print(f"Warning line {line_no}: initial value for {field} is being overwritten.")
                init_values[field] = convert_value(field, parts[1])

            elif len(parts) in (4, 5):
                field = canonical_field(parts[0])
                if field in INIT_ONLY_FIELDS:
                    raise ValueError(
                        f"Line {line_no}: {field} can only be set initially. "
                        "It cannot be changed by an animation command."
                    )

                try:
                    fi = int(parts[1])
                    ff = int(parts[2])
                except ValueError as exc:
                    raise ValueError(f"Line {line_no}: FI and FF must be integers: {cleaned}") from exc

                if fi < 0 or ff < 0:
                    raise ValueError(f"Line {line_no}: FI and FF must be non-negative: {cleaned}")
                if ff < fi:
                    raise ValueError(f"Line {line_no}: FF must be >= FI: {cleaned}")

                value = convert_value(field, parts[3])

                mode = "linear"
                if len(parts) == 5:
                    mode = parts[4].strip().lower()
                    if mode not in INTERPOLATION_MODES:
                        valid = ", ".join(sorted(INTERPOLATION_MODES))
                        raise ValueError(
                            f"Line {line_no}: unknown interpolation mode '{parts[4]}'. "
                            f"Valid modes: {valid}"
                        )

                if field in NONINTERPOLATED_FIELDS and mode != "linear":
                    print(
                        f"Warning line {line_no}: interpolation mode '{mode}' is ignored for {field}; "
                        f"{field} changes are always hard cuts."
                    )

                commands.append(CompilerCommand(field, fi, ff, value, mode, line_no, cleaned))

            else:
                raise ValueError(
                    f"Line {line_no}: expected 'VT VALUE', 'VT FI FF VF', or 'VT FI FF VF MODE', got: {cleaned}"
                )

    missing = sorted(REQUIRED_INITIAL_FIELDS - set(init_values.keys()))
    if missing:
        raise ValueError(
            "The command file is missing required initial values: " + ", ".join(missing)
        )

    state = WHCS.ViewerState()
    valid_state_fields = {f.name for f in fields(WHCS.ViewerState)}
    for field, value in init_values.items():
        if field not in valid_state_fields:
            raise ValueError(f"Internal error: {field} is not a ViewerState field.")
        setattr(state, field, value)

    state.snapping_enabled = False
    return state, commands


def frame_count_for_commands(commands: List[CompilerCommand]) -> int:
    if not commands:
        return 1

    # Normal interpolated commands affect frames [FI, FF), so FF defines the
    # length. Hard cuts with FI == FF still need frame FI to exist, so include
    # FI + 1 in that case.
    return max(max(cmd.ff, cmd.fi + 1 if cmd.fi == cmd.ff else cmd.ff) for cmd in commands)


def build_field_timelines(
    init_state: WHCS.ViewerState,
    commands: List[CompilerCommand],
) -> Dict[str, List[object]]:
    n_frames = frame_count_for_commands(commands)
    state_fields = {f.name for f in fields(WHCS.ViewerState)}

    timelines: Dict[str, List[object]] = {
        field: [copy.deepcopy(getattr(init_state, field)) for _ in range(n_frames)]
        for field in state_fields
    }
    timelines["snapping_enabled"] = [False for _ in range(n_frames)]

    commands_by_field: Dict[str, List[CompilerCommand]] = {}
    for cmd in commands:
        commands_by_field.setdefault(cmd.field, []).append(cmd)

    for field, field_commands in commands_by_field.items():
        field_commands = sorted(field_commands, key=lambda c: (c.fi, c.line_no))
        values = timelines[field]
        active_until = -1
        active_source = None

        for cmd in field_commands:
            if cmd.fi < active_until:
                print(
                    f"Warning line {cmd.line_no}: {field} command overlaps a previous command "
                    f"ending at frame {active_until}. The new command overwrites from frame {cmd.fi}.\n"
                    f"    previous: {active_source}\n"
                    f"    new:      {cmd.source}"
                )

            if field in NONINTERPOLATED_FIELDS:
                for frame_idx in range(cmd.fi, n_frames):
                    values[frame_idx] = cmd.value
                active_until = cmd.fi
                active_source = cmd.source
                continue

            start_value = values[cmd.fi]

            if cmd.ff == cmd.fi:
                for frame_idx in range(cmd.fi, n_frames):
                    values[frame_idx] = cmd.value
                active_until = cmd.fi
                active_source = cmd.source
                continue

            duration = cmd.ff - cmd.fi
            for frame_idx in range(cmd.fi, min(cmd.ff, n_frames)):
                alpha = (frame_idx - cmd.fi) / duration
                alpha = apply_interpolation_mode(alpha, cmd.mode)
                values[frame_idx] = start_value + (cmd.value - start_value) * alpha

            for frame_idx in range(cmd.ff, n_frames):
                values[frame_idx] = cmd.value

            active_until = cmd.ff
            active_source = cmd.source

    return timelines


def states_from_timelines(init_state: WHCS.ViewerState, timelines: Dict[str, List[object]]) -> List[WHCS.ViewerState]:
    n_frames = len(next(iter(timelines.values())))
    states: List[WHCS.ViewerState] = []
    state_fields = {f.name for f in fields(WHCS.ViewerState)}

    for frame_idx in range(n_frames):
        state = copy.deepcopy(init_state)
        for field in state_fields:
            value = timelines[field][frame_idx]
            if field in INTEGER_FIELDS:
                value = int(round(value))
            setattr(state, field, value)

        state.Xres = int(init_state.Xres)
        state.Yres = int(init_state.Yres)
        state.Tres = int(init_state.Tres)
        state.Cres = max(1, int(state.Cres))
        state.precision = max(1, int(state.precision))
        state.snapping_enabled = False
        state.current_frame = frame_idx
        states.append(state)

    return states


def validate_states_for_render(states: List[WHCS.ViewerState], mapping_path: str | None):
    for idx, state in enumerate(states):
        if state.indicator not in WHCS.INDICATORS:
            raise ValueError(f"Frame {idx}: unknown indicator {state.indicator}")
        if state.Cres < 1:
            raise ValueError(f"Frame {idx}: Cres must be >= 1")
        if mapping_path is not None and state.Cres > 24:
            raise ValueError(
                f"Frame {idx}: Cres={state.Cres}, but LUT mappings currently support at most 24 bits."
            )


def compile_command_file(path: str, mapping_path: str | None = None):
    init_state, commands = parse_command_file(path)
    timelines = build_field_timelines(init_state, commands)
    states = states_from_timelines(init_state, timelines)
    validate_states_for_render(states, mapping_path)
    return init_state, states


def main():
    ap = argparse.ArgumentParser(description="Compile a WHCS command file into an MP4 animation.")
    default_command_file = SCRIPT_DIR / "commandSets" / "mandelCmds.txt"
    default_output_file = SCRIPT_DIR / "outputs" / "frac_cmded.mp4"

    ap.add_argument(
        "command_file",
        nargs="?",
        default=str(default_command_file),
        help="Path to the WHCS command file.",
    )

    ap.add_argument(
        "--output",
        default=str(default_output_file),
        help="Output .mp4 path.",
    )
    ap.add_argument("--mapping", default=None, help="Optional .npz mapping file. If provided, LUT color output is used.")
    ap.add_argument("--fps", type=int, default=30, help="Output video frame rate.")
    ap.add_argument(
        "--display",
        type=int,
        default=0,
        help="Preview every Nth rendered frame. Use --display 1 for every frame. Default 0 disables preview.",
    )
    args = ap.parse_args()

    if args.display < 0:
        raise ValueError("--display must be 0 or a positive integer.")
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")

    init_state, frame_states = compile_command_file(args.command_file, mapping_path=args.mapping)

    print(f"Compiled {len(frame_states)} output frames.")
    print(f"Initial indicator: {init_state.indicator}")
    print(f"Output: {os.path.abspath(args.output)}")
    if args.mapping:
        print(f"Using LUT mapping: {os.path.abspath(args.mapping)}")

    WHCS.generateFromCode(
        init_state=init_state,
        frame_states=frame_states,
        output_path=args.output,
        mapping_path=args.mapping,
        fps=args.fps,
        display_every=args.display,
    )


if __name__ == "__main__":
    main()
