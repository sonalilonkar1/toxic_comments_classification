import os, zipfile, pathlib, subprocess

def main():
    ROOT = pathlib.Path(__file__).resolve().parents[2]
    DATA = ROOT / "data" / "raw"
    DATA.mkdir(parents=True, exist_ok=True)

    # Prefer project-root kaggle.json if present; else use ~/.kaggle
    if (ROOT / "kaggle.json").exists():
        os.environ["KAGGLE_CONFIG_DIR"] = str(ROOT)
    else:
        # ensure default exists
        kag = pathlib.Path.home() / ".kaggle" / "kaggle.json"
        if not kag.exists():
            raise SystemExit("kaggle.json not found. Put it in project root or ~/.kaggle/")
    print("Using KAGGLE_CONFIG_DIR =", os.environ.get("KAGGLE_CONFIG_DIR", "~/.kaggle"))

    # ensure kaggle CLI exists
    try:
        subprocess.run(["kaggle","--version"], check=True, capture_output=True)
    except Exception as e:
        raise SystemExit("Kaggle CLI not found. Install with: pip install kaggle") from e

    print("Downloading Jigsaw Toxic Comment Challenge data...")
    subprocess.run([
        "kaggle","competitions","download",
        "-c","jigsaw-toxic-comment-classification-challenge",
        "-p", str(DATA)
    ], check=True)

    for z in DATA.glob("*.zip"):
        print("Unzipping:", z.name)
        with zipfile.ZipFile(z, "r") as f:
            f.extractall(DATA)
    print("Done. Raw files are in:", DATA)

if __name__ == "__main__":
    main()
