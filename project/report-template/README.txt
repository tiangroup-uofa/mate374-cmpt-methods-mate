ACS-style sample report

Replace the blue placeholders and example figures with the group's own work.

In the course checkout, compile report.tex from this folder with:
  latexmk -pdf report.tex
It reads the root ../../references.bib. Add or correct bibliography entries there.

scripts/package_project.py builds the private Canvas package. Its portable source ZIP
changes the bibliography path and includes a bibliography copy made from the root file.
Students can upload that complete source ZIP to Overleaf.

Do not put student names or the group assignment PDF in this public source folder.
