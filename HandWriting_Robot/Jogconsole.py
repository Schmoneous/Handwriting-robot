#!/usr/bin/env python3
"""
jog_console.py
===============
Interactive, safe jogging tool for an EBB-based board (like your DrawCore V1.2)
to help you find the physical XY travel bounds.
 
How it works:
  - You jog in small steps using simple text commands.
  - After every single move, it queries the board's real position (QS) and
    prints it, plus tracks the min/max ever reached on each axis.
  - You watch the machine physically and stop BEFORE it hits a hard limit --
    back off a little as a safety margin.
  - At the end, it prints a summary of the bounds you found.
 
Usage:
    python3 jog_console.py --port /dev/tty.usbmodem11401
 
Commands once running (type one, press Enter):
    x+          jog axis1 (X) positive by <step>
    x-          jog axis1 (X) negative by <step>
    y+          jog axis2 (Y) positive by <step>
    y-          jog axis2 (Y) negative by <step>
    step 200    change the jog step size (in motor steps) to 200
    zero        tell the board to treat current position as (0,0) -- uses CS
    pos         just query and print current position (no movement)
    home        (optional) attempt to return to (0,0) -- ONLY use after you've
                confirmed (0,0) is a safe, reachable point
    q           quit and print the bounds summary
 
Safety notes:
  - Start with a SMALL step size (default 100 steps) especially near suspected
    edges. Increase it only in open, confirmed-safe space.
  - If you hear rattling/skipping/grinding, STOP immediately (q) -- that means
    it's already hit a hard stop and is skipping steps. The reported QS
    position will no longer be trustworthy after a skip, so back off physically
    before continuing to jog.
  - This script enables motors on startup (EM,1,1). Make sure the machine is
    powered and nothing is in the way before running it.
"""
 
import argparse
import sys
import time
 
import serial
 
 
def send_cmd(ser, cmd, verbose=True):
    """Send one command, return the raw response lines (until OK or error)."""
    ser.write((cmd + "\r").encode())
    time.sleep(0.05)
    lines = []
    deadline = time.time() + 3
    while time.time() < deadline:
        line = ser.readline().decode(errors="replace").strip()
        if not line:
            if lines:
                break
            continue
        lines.append(line)
        if line.upper().startswith("OK") or "err" in line.lower():
            break
    if verbose:
        for l in lines:
            print(f"    < {l}")
    return lines
 
 
def query_position(ser):
    """Send QS and parse out the two axis step counts."""
    lines = send_cmd(ser, "QS", verbose=False)
    for l in lines:
        parts = l.replace(" ", "").split(",")
        if len(parts) == 2:
            try:
                a, b = int(parts[0]), int(parts[1])
                return a, b
            except ValueError:
                continue
    return None, None
 
 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True, help="Serial port, e.g. /dev/tty.usbmodem11401")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--step", type=int, default=100, help="Initial jog step size (motor steps).")
    p.add_argument("--duration-ms", type=int, default=800, help="Time each jog move takes.")
    args = p.parse_args()
 
    print(f"[info] connecting to {args.port} @ {args.baud} ...")
    ser = serial.Serial(args.port, args.baud, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()
 
    print("[info] enabling motors (EM,1,1) ...")
    send_cmd(ser, "EM,1,1")
 
    step = args.step
    duration = args.duration_ms
 
    x_pos = y_pos = 0
    x_min = y_min = 0
    x_max = y_max = 0
 
    a0, b0 = query_position(ser)
    if a0 is not None:
        x_pos, y_pos = a0, b0
        x_min = x_max = x_pos
        y_min = y_max = y_pos
    print(f"[info] starting position: X={x_pos} Y={y_pos}")
    print(f"[info] jog step = {step} steps, move duration = {duration} ms")
    print(__doc__.split("Commands once running")[1].split("Safety notes")[0])
 
    while True:
        try:
            raw = input(f"[X={x_pos} Y={y_pos} | step={step}] jog> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
 
        if not raw:
            continue
 
        parts = raw.split()
        cmd = parts[0].lower()
 
        if cmd == "q":
            break
 
        elif cmd == "pos":
            a, b = query_position(ser)
            if a is not None:
                x_pos, y_pos = a, b
            print(f"    X={x_pos}  Y={y_pos}")
 
        elif cmd == "step" and len(parts) == 2:
            try:
                step = int(parts[1])
                print(f"    step size now {step}")
            except ValueError:
                print("    usage: step <integer>")
 
        elif cmd == "zero":
            send_cmd(ser, "CS")
            x_pos = y_pos = 0
            x_min = x_max = 0
            y_min = y_max = 0
            print("    position counters reset to (0,0)")
 
        elif cmd == "home":
            print("    homing to (0,0) via SM ...")
            # Simple relative move back to zero based on tracked position.
            dx, dy = -x_pos, -y_pos
            dur = max(300, min(8000, int(max(abs(dx), abs(dy)) * 2)))
            send_cmd(ser, f"SM,{dur},{dx},{dy}")
            a, b = query_position(ser)
            if a is not None:
                x_pos, y_pos = a, b
 
        elif cmd in ("x+", "x-", "y+", "y-"):
            sign = 1 if cmd.endswith("+") else -1
            if cmd.startswith("x"):
                dx, dy = sign * step, 0
            else:
                dx, dy = 0, sign * step
            send_cmd(ser, f"SM,{duration},{dx},{dy}")
            a, b = query_position(ser)
            if a is not None:
                x_pos, y_pos = a, b
                x_min, x_max = min(x_min, x_pos), max(x_max, x_pos)
                y_min, y_max = min(y_min, y_pos), max(y_max, y_pos)
 
        else:
            print("    unknown command. Use x+ x- y+ y- step <n> zero pos home q")
 
    print("\n[summary]")
    print(f"  X range explored: {x_min} .. {x_max}  (span {x_max - x_min} steps)")
    print(f"  Y range explored: {y_min} .. {y_max}  (span {y_max - y_min} steps)")
    print("  NOTE: these are only the bounds YOU jogged to, not necessarily the")
    print("  true mechanical limits. Add a safety margin (5-10%) before using")
    print("  these numbers as soft limits in the handwriting generator.")
 
    ser.close()
 
 
if __name__ == "__main__":
    main()
 