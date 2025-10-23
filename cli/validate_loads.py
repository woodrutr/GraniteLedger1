import argparse
import sys
import json

from granite_io.frames_api import build_frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", required=True)
    p.add_argument("--scenario", required=True)
    p.add_argument("--isos", nargs="*", default=[])
    args = p.parse_args()

    sel = {"load": {"scenario": args.scenario}}
    if args.isos:
        sel["load"]["isos"] = args.isos

    try:
        frames = build_frames(args.input_root, sel)
        df = frames.load
        problems = []

        if df["load_gwh"].le(0).any():
            problems.append("non-positive load_gwh found")
        if df["year"].duplicated().any() and df[["iso", "zone", "year"]].duplicated().any():
            problems.append("duplicate iso-zone-year rows")

        print(
            json.dumps(
                {
                    "rows": int(df.shape[0]),
                    "isos": sorted(df["iso"].unique().tolist()),
                    "zones": int(df["zone"].nunique()),
                    "years": f"{int(df['year'].min())}-{int(df['year'].max())}",
                    "problems": problems,
                },
                indent=2,
            )
        )
        sys.exit(0 if not problems else 2)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
