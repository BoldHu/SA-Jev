import json
import os
from pathlib import Path
import subprocess
import sys

from sa_jev.environment import CREDENTIAL_KEYS, load_environment


def clean_credentials(monkeypatch):
    for key in (*CREDENTIAL_KEYS, "SAJEV_ENV_FILE"):
        monkeypatch.delenv(key, raising=False)


def test_env_reload_quotes_bom_and_blank_values(tmp_path, monkeypatch, capsys):
    clean_credentials(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "old-session-value")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://old.example/v1")
    monkeypatch.setenv("JEV_API_KEY", "old-legacy-value")
    path = tmp_path / ".env"
    token = "test-${LITERAL} # token"
    path.write_text(f"OPENAI_API_KEY='{token}'\nOPENAI_BASE_URL=\nHF_TOKEN=\n", encoding="utf-8-sig")
    status = load_environment(path)
    assert os.environ["OPENAI_API_KEY"] == token
    assert "OPENAI_BASE_URL" not in os.environ
    assert "JEV_API_KEY" not in os.environ
    assert status["OPENAI_API_KEY"] and not status["HF_TOKEN"]
    path.write_text("OPENAI_API_KEY=\nTYPESAFE_API_KEY=test-new-jev\n", encoding="utf-8")
    load_environment(path)
    assert "OPENAI_API_KEY" not in os.environ
    assert os.environ["TYPESAFE_API_KEY"] == "test-new-jev"
    assert capsys.readouterr().out == ""


def test_all_notebooks_share_drive_file_and_preserve_filled_values(tmp_path, monkeypatch, capsys):
    clean_credentials(monkeypatch)
    repo = Path(__file__).resolve().parents[1]
    project, drive = tmp_path / "project", tmp_path / "drive"
    project.mkdir()
    (project / ".env.example").write_bytes((repo / ".env.example").read_bytes())
    monkeypatch.setitem(sys.modules, "google.colab", object())
    for index, notebook in enumerate(sorted((repo / "notebooks").glob("*.ipynb"))):
        nb = json.loads(notebook.read_text(encoding="utf-8"))
        cell = [c for c in nb["cells"] if c["cell_type"] == "code"][2]
        namespace = {"PROJECT": project, "DRIVE": drive, "sys": sys}
        exec("".join(cell["source"]), namespace)
        assert namespace["ENV_FILE"] == drive / ".env"
        if index == 0:
            (drive / ".env").write_text("OPENAI_API_KEY=test-drive-key\nHF_TOKEN=test-hf-key\n", encoding="utf-8")
        else:
            assert os.environ["OPENAI_API_KEY"] == "test-drive-key"
            assert os.environ["HF_TOKEN"] == "test-hf-key"
    output = capsys.readouterr().out
    assert "test-drive-key" not in output and "test-hf-key" not in output
    assert "loaded" in output and "empty" in output


def test_cli_child_reuses_drive_file_and_legacy_jev_alias(tmp_path):
    path = tmp_path / ".env"
    path.write_text("OPENAI_API_KEY=test-cli-key\nJEV_API_KEY=test-legacy-key\n", encoding="utf-8")
    script = """
import os, sys, types
from sa_jev.cli import main
def run(root, device, raw):
    assert os.environ['OPENAI_API_KEY'] == 'test-cli-key'
    assert os.environ['TYPESAFE_API_KEY'] == 'test-legacy-key'
    assert 'OPENAI_BASE_URL' not in os.environ
    return 'CLI credential loading passed'
smoke = types.ModuleType('sa_jev.smoke')
smoke.run = run
sys.modules['sa_jev.smoke'] = smoke
sys.argv = ['sajev', 'smoke']
main()
"""
    env = dict(os.environ, SAJEV_ENV_FILE=str(path), OPENAI_API_KEY="stale-key", OPENAI_BASE_URL="https://old.example/v1")
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True)
    assert "CLI credential loading passed" in result.stdout
    assert "test-cli-key" not in result.stdout and "test-legacy-key" not in result.stdout
