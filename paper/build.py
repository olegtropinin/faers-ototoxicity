"""Inline generated tables (\\input inside tabular breaks \\multicolumn) and compile with tectonic."""
import re
import subprocess
from pathlib import Path

here = Path(__file__).parent
src = (here / "main.tex").read_text()
full = re.sub(r"\\input\{(generated/[^}]+)\}", lambda m: (here / m.group(1)).read_text().rstrip("\n"), src)
(here / "manuscript.tex").write_text(full)
subprocess.run(["tectonic", "-X", "compile", "manuscript.tex"], cwd=here, check=True)
