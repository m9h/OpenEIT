#!/usr/bin/env python
"""First-contact probe for the Spectra over USB serial (115200 8N1).

Listens only by default; pass --mode to send a firmware mode letter
(a=time series, b=BIS, c/d/e = 8/16/32-electrode EIT).

  python tools/spectra_probe.py                 # auto-detect port, listen 10 s
  python tools/spectra_probe.py --mode c -t 20  # 8-electrode EIT for 20 s
"""
import argparse
import sys
import time

import numpy as np
import serial
import serial.tools.list_ports

EXPECTED = {32: 8, 192: 16, 896: 32}  # opposition-pattern frame lengths


def find_port():
    ports = [p for p in serial.tools.list_ports.comports()
             if p.vid == 0x0403 or "usbserial" in p.device]
    return ports[0].device if ports else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?")
    ap.add_argument("-t", "--seconds", type=float, default=10)
    ap.add_argument("--mode", choices=list("abcdefg"))
    ap.add_argument("-o", "--out", help="append raw lines to this file")
    args = ap.parse_args()

    port = args.port or find_port()
    if not port:
        sys.exit("No FTDI serial port found. Board unpowered (switch), charge-only cable, or hub port.")
    print("port:", port)

    frames = []
    with serial.Serial(port, 115200, timeout=1) as ser:
        if args.mode:
            ser.write((args.mode + "\n").encode())
            print("sent mode", args.mode)
        out = open(args.out, "a") if args.out else None
        end = time.time() + args.seconds
        while time.time() < end:
            line = ser.readline().decode(errors="replace").strip()
            if not line:
                continue
            if out:
                out.write(line + "\n")
            if ":" in line:
                try:
                    vals = [float(v) for v in line.split(":", 1)[1].replace(";", ",").split(",") if v.strip()]
                except ValueError:
                    vals = []
                if len(vals) > 1:
                    frames.append((time.time(), vals))
                    print(f"frame {len(frames):3d}: {len(vals)} values  "
                          f"min {min(vals):.4g}  median {np.median(vals):.4g}  max {max(vals):.4g}")
                    continue
            print("text:", line[:120])
        if out:
            out.close()

    if not frames:
        sys.exit("Port opened but no data frames arrived.")
    lens = {len(v) for _, v in frames}
    print("\nframes:", len(frames), "lengths:", lens)
    if len(frames) > 1:
        dt = np.diff([t for t, _ in frames])
        print(f"frame period: {np.median(dt):.2f} s")
    n = max(lens, key=lambda k: sum(len(v) == k for _, v in frames))
    data = np.array([v for _, v in frames if len(v) == n])
    if n in EXPECTED:
        print(f"{EXPECTED[n]}-electrode EIT frames")
    zeros = (data == 0).mean(axis=0)
    print(f"measurements reading zero (open circuit) in every frame: {(zeros == 1).sum()} of {n}")
    if len(data) > 2:
        mean = data.mean(axis=0)
        cv = data.std(axis=0) / np.where(mean == 0, np.nan, np.abs(mean))
        print(f"frame-to-frame variation: median {np.nanmedian(cv) * 100:.2f}%  worst {np.nanmax(cv) * 100:.1f}%")


if __name__ == "__main__":
    main()
