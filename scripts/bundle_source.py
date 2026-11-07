"""Create a source-only Colab archive from an explicit allowlist; never include secrets/data/weights."""
from pathlib import Path
import zipfile

root=Path(__file__).resolve().parents[1]
target=root/'artifacts/sa-jev-source.zip'; target.parent.mkdir(exist_ok=True)
files=[root/n for n in ('AGENTS.md','README.md','pyproject.toml','environment.yml','.gitignore','.env.example')]
for folder in ('src/sa_jev','scripts','tests','configs','notebooks','docs'):
    files.extend(p for p in (root/folder).rglob('*') if p.is_file() and p.suffix in {'.py','.md','.json','.yaml','.yml','.ipynb','.txt'} and '__pycache__' not in p.parts)
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(set(files)):
        if p.exists(): z.write(p,p.relative_to(root))
print(target, target.stat().st_size, 'bytes')
