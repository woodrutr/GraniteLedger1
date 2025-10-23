from pathlib import Path
import shutil

def main(project_root: str = "."):
    root = Path(project_root)
    units_dir = root / "input" / "engine" / "units"
    old = units_dir / "us_units.csv"
    new = units_dir / "ei_units.csv"
    if old.exists() and not new.exists():
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old, new)
        print("Copied", old, "->", new)
    else:
        print("No action. Old exists:", old.exists(), "New exists:", new.exists())

if __name__ == "__main__":
    main()
