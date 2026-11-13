ACS-style sample report

Replace the blue placeholders and example figures with the group's own work.

In the course checkout, compile report.tex from this folder with:
  latexmk -pdf report.tex
It reads the root ../../references.bib. Add or correct bibliography entries there.

scripts/package_project.py builds the private Canvas package. Its portable source ZIP
changes the bibliography path and includes a bibliography copy made from the root file.
Students can use that complete source ZIP in their LaTeX editor, including Overleaf.

The public website download is project/MATE374-group-project.zip. Build it with
scripts/package_project.py --public --assignment-pdf PATH_TO_ROSTER_FREE_PDF
--output project/MATE374-group-project.zip after rendering the current PDFs.
The public assignment must omit the student group roster. The named Canvas package
and authoring sources stay under ignored instructor-private/.

Do not put student names or the group assignment PDF in this public source folder.
