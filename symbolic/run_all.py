"""Run every symbolic verification script and report a summary table.

Usage:  python run_all.py [--fast]

--fast runs only the quick scripts (skips the three heaviest pipelines);
use it as a smoke test. The full run takes ~15 minutes on a laptop.
Exit code 0 iff every script passed.
"""

import argparse
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent

SCRIPTS = [
    # (script, heavy)
    ("tests_averaging.py", False),
    ("check_form_factors.py", False),
    ("check_steady_states.py", False),
    ("check_kick_covariance.py", False),
    ("check_tidal_adiabatic.py", False),
    ("check_tidal_whitenoise.py", True),
    ("check_tidal_traceless_gw.py", True),
    ("check_impulsive_general.py", False),
    ("check_impulsive_tidal_limit.py", False),
    ("check_impulsive_point_mass.py", True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="skip the heaviest scripts (smoke test)")
    args = ap.parse_args()

    results = []
    for script, heavy in SCRIPTS:
        if args.fast and heavy:
            results.append((script, "SKIPPED", 0.0))
            continue
        t0 = time.time()
        proc = subprocess.run([sys.executable, "-u", str(HERE / script)],
                              capture_output=True, text=True)
        dt = time.time() - t0
        status = "PASS" if proc.returncode == 0 else "FAIL"
        results.append((script, status, dt))
        print(f"[{status}] {script}  ({dt:.0f}s)")
        if status == "FAIL":
            print(proc.stdout[-3000:])
            print(proc.stderr[-2000:])

    print("\n" + "=" * 56)
    n_fail = 0
    for script, status, dt in results:
        print(f"  {script:38s} {status:8s} {dt:6.0f}s")
        n_fail += status == "FAIL"
    print("=" * 56)
    print(f"{n_fail} failures out of {len(results)} scripts")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
