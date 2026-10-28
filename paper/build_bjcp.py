"""BJCP variant of the manuscript, from the files build.py and build_docx.py write: Aim/Methods/Results/Conclusion
abstract headings, keywords, and BJCP's 'What is already known' / 'What this study adds' statements in place of key points."""
import subprocess
from pathlib import Path

here = Path(__file__).parent
POINTS = (here / "generated/bjcp_points.tex").read_text()
KEYWORDS = (r"\noindent\textbf{Keywords:} adverse drug reaction reporting systems; pharmacovigilance; hearing loss; "
            "tinnitus; ototoxicity; signal detection\n")


def adapt(s):
    s = s[:s.index(r"\section*{Key points}")] + POINTS + "\n\n" + s[s.index(r"\section{Introduction}"):]
    for old, new in ((r"\textbf{Introduction:}", r"\textbf{Aim:}"), (r"\textbf{Conclusions:}", r"\textbf{Conclusion:}"),
                     (r"\end{abstract}", r"\end{abstract}" + "\n" + KEYWORDS)):
        assert old in s, old
        s = s.replace(old, new, 1)
    return s


(here / "manuscript_bjcp.tex").write_text(adapt((here / "manuscript.tex").read_text()))
(here / "manuscript_bjcp_docx.tex").write_text(adapt((here / "manuscript_docx.tex").read_text()))
subprocess.run(["tectonic", "-X", "compile", "manuscript_bjcp.tex"], cwd=here, check=True)
subprocess.run(["pandoc", "manuscript_bjcp_docx.tex", "-o", "manuscript_bjcp.docx", "--citeproc", "--bibliography=refs.bib",
                "--csl=springer-vancouver-brackets.csl", "--resource-path=.:..:../figures"], cwd=here, check=True)
print("manuscript_bjcp.pdf, manuscript_bjcp.docx")
