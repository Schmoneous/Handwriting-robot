#!/usr/bin/env python3
"""
test_switch.py
================
Quick standalone test for a single EBB digital input pin -- no GUI, no
homing logic, just live pin state so you can confirm a switch (mechanical
or hall-effect) is wired and working before trusting it in the real
homing routine.

Usage:
    python3 test_switch.py --port /dev/tty.usbmodem11401 --pin B7

    (defaults to B7, active-low, matching the hall-effect Y sensor --
    pass --pin B5 or --pin B6 to test the mechanical X switches instead,
    which are active-high, e.g.:
        python3 test_switch.py --port /dev/tty.usbmodem11401 --pin B5 --active-high)

Prints a line only when the state CHANGES (not on every poll), so you get
a clean log as you move the magnet/press the switch by hand, e.g.:

    [12:03:41.221] B7 = 1 (idle)
    [12:03:44.009] B7 = 0 (TRIGGERED)
    [12:03:46.552] B7 = 1 (idle)

Press Ctrl+C to stop.
"""

import argparse
import time

import serial


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True, help="Serial port, e.g. /dev/tty.usbmodem11401")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--pin", default="B7", help="Port+pin, e.g. B7, B5, B6 (default: B7)")
    p.add_argument("--active-high", action="store_true",
                    help="Treat '1' as triggered instead of the default '0' "
                         "(use this for the mechanical X switches, B5/B6 -- "
                         "the hall-effect Y sensor on B7 is active-low by default)")
    p.add_argument("--poll-ms", type=int, default=20, help="How often to check, in ms (default 20)")
    args = p.parse_args()

    port_letter = args.pin[0].upper()
    pin_number = args.pin[1:]
    triggered_value = "1" if args.active_high else "0"

    print(f"[info] connecting to {args.port} @ {args.baud} ...")
    ser = serial.Serial(args.port, args.baud, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()

    print(f"[info] configuring {port_letter}{pin_number} as input (PD)...")
    ser.write(f"PD,{port_letter},{pin_number},1\r".encode())
    resp = ser.readline().decode(errors="replace").strip()
    print(f"    < {resp}")

    print(f"[info] watching {port_letter}{pin_number}  "
          f"(triggered = '{triggered_value}')  -- press Ctrl+C to stop\n")

    last_state = None
    try:
        while True:
            ser.reset_input_buffer()
            ser.write(f"PI,{port_letter},{pin_number}\r".encode())
            line = ser.readline().decode(errors="replace").strip()

            value = None
            if line.startswith("PI,"):
                value = line.split(",")[1]

            if value is not None and value != last_state:
                ts = time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"
                label = "TRIGGERED" if value == triggered_value else "idle"
                print(f"[{ts}] {port_letter}{pin_number} = {value} ({label})")
                last_state = value

            time.sleep(args.poll_ms / 1000.0)

    except KeyboardInterrupt:
        print("\n[info] stopped.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()