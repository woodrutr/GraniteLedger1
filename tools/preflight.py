import subprocess
import sys

steps = [
    ["python", "tools/validate_regions.py"],
    ["python", "-m", "engine.io.load_forecasts_strict", "--dry-run"],  # add argparse stub to not write parquet
]

for cmd in steps:
    print("+", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc:
        sys.exit(rc)

print("OK")
