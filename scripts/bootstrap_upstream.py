"""Fetch only source; apply the audited 255 -> 256 interface limit patch."""
from pathlib import Path
import subprocess

REVISION = "e279c44829370cbdc5c48784a912f38ffd73df60"


def bootstrap(root=None):
    root = Path(root or Path(__file__).resolve().parents[1])
    target = root / "vendor" / "OpenJev"
    if not target.exists():
        subprocess.run(["git", "clone", "https://github.com/ejhshen/OpenJev.git", str(target)], check=True)
    current = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
    if current != REVISION:
        subprocess.run(["git", "-C", str(target), "checkout", REVISION], check=True)
    for relative in ("src/openjev/models/decision/schema.py", "src/openjev/api/typesafe.py"):
        path = target / relative
        source = path.read_text(encoding="utf-8")
        if "<= 255" not in source and "<= 256" not in source:
            raise RuntimeError("upstream limit changed; inspect before patching")
        source = source.replace("<= 255", "<= 256").replace("1 to 255", "1 to 256")
        path.write_text(source, encoding="utf-8")
    print(f"OpenJev source {REVISION}; only option-count validation extended to 256")


if __name__ == "__main__":
    bootstrap()
