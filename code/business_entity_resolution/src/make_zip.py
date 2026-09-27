"""Build <team>_submission.zip in the layout the challenge asks for.

    python src/make_zip.py --team Vrishabh
"""
import argparse
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)                      # code/business_entity_resolution
ROOT = os.path.dirname(os.path.dirname(PROJECT))     # student_resource


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    args = ap.parse_args()

    files = [
        ("output/matching_results.tsv", os.path.join(ROOT, "output", "matching_results.tsv")),
        ("output/candidate_pairs.tsv", os.path.join(ROOT, "output", "candidate_pairs.tsv")),
        ("Documentation_template.md", os.path.join(ROOT, "Documentation_template.md")),
        ("code/business_entity_resolution/README.md", os.path.join(PROJECT, "README.md")),
        ("code/business_entity_resolution/requirements.txt", os.path.join(PROJECT, "requirements.txt")),
    ]
    for name in sorted(os.listdir(HERE)):
        if name.endswith(".py") and not name.startswith("_"):
            files.append((f"code/business_entity_resolution/src/{name}", os.path.join(HERE, name)))

    out = os.path.join(ROOT, f"{args.team}_submission.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for arc, path in files:
            z.write(path, arc)
            print(" ", arc)
    print(f"-> {out} ({os.path.getsize(out) / 2**20:.0f} MB)")


if __name__ == "__main__":
    main()
