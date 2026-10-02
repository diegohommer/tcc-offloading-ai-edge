# Project instructions

## The thesis is mirrored to Overleaf — pull before you edit

`thesis/latex/` is synced with the Overleaf project "TCC Diego Amorim" through
[olcli](https://github.com/aloth/olcli). **The advisor edits there**, so Overleaf can be
ahead of this checkout at any moment. Treat it exactly as you would a shared git remote.

**Before changing anything under `thesis/latex/`:**

```bash
cd thesis/latex && olcli pull     # bring the advisor's edits down
git diff                          # see exactly what he changed
```

Commit his changes before adding your own, so his work is in history and never mixed into
one of your commits.

**After changing anything under `thesis/latex/`:**

```bash
cd thesis/latex
touch <the files you edited>      # push selects by mtime, see below
olcli diff                        # preview what would reach Overleaf
olcli push                        # upload only the changed files
```

### Rules

1. **Never run `olcli sync`.** It resolves conflicts by timestamp ("local wins if newer"),
   which silently overwrites the advisor's edits. Pull, diff and push only.
2. **`olcli push` picks files by modification time, not content.** A file edited before the
   last pull looks unchanged to it and is skipped. `touch` it, or confirm with
   `olcli push --dry-run`, which lists what would be sent.
3. **Never push without pulling first.** A push uploads your version of a changed file; it
   does not merge.
4. **`thesis/latex/.olignore`** keeps local-only helpers (`watch.sh`) off Overleaf. LaTeX
   build artifacts are filtered by olcli's own default list.

### If olcli reports zero projects

The session cookie is fine; Overleaf is redirecting every request to a "confirm your primary
email" interstitial. Ask the user to open overleaf.com and answer it, then retry. Re-auth
with `olcli auth --cookie "s%3A..."` only if `olcli whoami` actually fails.

`olcli comments` lists the Overleaf review threads, for when the advisor comments instead of
editing.

## Building the thesis

Build into `/tmp/claude-build`, never in `thesis/latex`, because the user's editor
recompiles there on every save and the two builds corrupt each other's aux files:

```bash
cd thesis/latex && pdflatex -interaction=nonstopmode -output-directory=/tmp/claude-build tcc.tex
```

Run `bibtex` inside that directory with `BIBINPUTS` and `BSTINPUTS` pointing at
`thesis/latex`.
