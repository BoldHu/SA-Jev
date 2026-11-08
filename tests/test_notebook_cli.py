import ast
import json
from pathlib import Path
import subprocess
import sys
import pytest


def test_notebook_cli_exposes_child_error_and_success_output(tmp_path, capsys):
    repo = Path(__file__).resolve().parents[1]
    nb = json.loads((repo / "notebooks/00_environment_data.ipynb").read_text(encoding="utf-8"))
    install = [c for c in nb["cells"] if c["cell_type"] == "code"][1]
    tree = ast.parse("".join(install["source"]))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "cli")
    namespace = {"ROOT": tmp_path, "CONFIG_PATH": repo/"configs/public_lightrag.yaml",
                 "sys": sys, "subprocess": subprocess}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "notebook-cli", "exec"), namespace)
    with pytest.raises(RuntimeError, match="invalid choice") as error:
        namespace["cli"]("unknown-command-for-test")
    assert "exit 2" in str(error.value)
    assert "invalid choice" in capsys.readouterr().out
    namespace["cli"]("--help")
    assert "usage:" in capsys.readouterr().out
