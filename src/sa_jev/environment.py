"""Load service credentials from one local or Drive .env file without logging values."""
from pathlib import Path
import os
import sys
from dotenv import dotenv_values


CREDENTIAL_KEYS = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "TYPESAFE_API_KEY", "HF_TOKEN", "JEV_API_KEY")


def load_environment(path, *, template=None):
    """The file owns these settings; blank/omitted values clear stale session values."""
    path = Path(path).expanduser().resolve()
    if not path.exists() and template is not None:
        content = Path(template).read_bytes()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation preserves a file the user has already filled in.
        with path.open("xb") as f:
            f.write(content)
    if not path.is_file():
        raise FileNotFoundError(f"Fill in the credential file first: {path}")
    values = dotenv_values(path, encoding="utf-8-sig", interpolate=False)
    for key in CREDENTIAL_KEYS:
        value = (values.get(key) or "").strip()
        if value:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)
    # Keep the older Jev variable name compatible with all notebook baselines.
    if not os.environ.get("TYPESAFE_API_KEY") and os.environ.get("JEV_API_KEY"):
        os.environ["TYPESAFE_API_KEY"] = os.environ["JEV_API_KEY"]
    os.environ["SAJEV_ENV_FILE"] = str(path)
    return {key: bool(os.environ.get(key)) for key in CREDENTIAL_KEYS}


def disconnect_after_preparation(root, *, run_id, enabled=True):
    """Flush Drive, then release Colab only after this run's preparation succeeded."""
    if not enabled:
        print("Automatic disconnect disabled.", flush=True)
        return False
    if "google.colab" not in sys.modules:
        print("Not running in Colab; automatic disconnect skipped.", flush=True)
        return False
    from .io import read_json
    marker = read_json(Path(root) / "00_complete.json")
    if not run_id or marker.get("run_id") != run_id or marker.get("status") != "data_preparation_complete":
        raise RuntimeError("This run has not completed preparation; keeping the runtime connected.")
    from google.colab import drive, runtime
    print("00 complete. Flushing Drive writes before releasing this runtime...", flush=True)
    # Do not release the VM if flushing fails. Transient /content files are not retained.
    drive.flush_and_unmount()
    print("Drive flushed. Releasing the Colab runtime.", flush=True)
    runtime.unassign()
    return True
