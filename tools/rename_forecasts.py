import os
from pathlib import Path

# Define the root directory for load forecasts
FORECAST_DIR = Path("input/electricity/load_forecasts")

# Define the specific renames to be made
RENAMES = {
    "southeast/southeast_2024_baseline/southeast_frcc.csv": "southeast/southeast_2024_baseline/FRCC_SYS.csv",
    "southeast/southeast_2024_baseline/southeast_santee.csv": "southeast/southeast_2024_baseline/SANTEE_COOPER_SYS.csv",
}

def main():
    """Renames specific forecast files to their canonical region_id."""
    if not FORECAST_DIR.is_dir():
        print(f"ERROR: Directory not found: {FORECAST_DIR}")
        return

    print("Applying file renames...")
    for old_rel_path, new_rel_path in RENAMES.items():
        old_file = FORECAST_DIR / old_rel_path
        new_file = FORECAST_DIR / new_rel_path

        if old_file.exists():
            print(f"  RENAME: {old_file} -> {new_file}")
            old_file.rename(new_file)
        else:
            print(f"  SKIPPED: Source file not found: {old_file}")

    print("Done.")

if __name__ == "__main__":
    main()
