"""Word version for submission (Drug Safety asks for Word): strip LaTeX-only wrappers, then pandoc with Springer Vancouver."""
import re
import subprocess
from pathlib import Path

here = Path(__file__).parent
s = (here / "manuscript.tex").read_text()
s = re.sub(r"\\resizebox\{\\textwidth\}\{!\}\{%\n(\\begin\{tabular\}.*?\\end\{tabular\})\}", r"\1", s, flags=re.S)
s = s.replace(r"$\sum_i P(X\geq a_{\mathrm{crit},i}\mid E_i \times \mathrm{OR}_i)$",
              r"the sum over ingredients of the Poisson probability of reaching $a_{crit}$ when the mean is $E_i$ times OR$_i$")
(here / "manuscript_docx.tex").write_text(s)
subprocess.run(["pandoc", "manuscript_docx.tex", "-o", "manuscript.docx", "--citeproc", "--bibliography=refs.bib",
                "--csl=springer-vancouver-brackets.csl", "--resource-path=.:..:../figures"], cwd=here, check=True)
subprocess.run(["pandoc", "supplement.tex", "-o", "supplement.docx", "--resource-path=.:..:../figures"], cwd=here, check=True)
print("manuscript.docx, supplement.docx")
