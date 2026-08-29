#!/usr/bin/env bash
# One-time setup: create the repo, push, and turn on GitHub Pages.
# Run this after `gh auth login`.
set -euo pipefail

USER="${1:-sgpfaff}"
REPO="${2:-tambora-docs}"

gh repo create "$USER/$REPO" --public --source=. --remote=origin \
  --description "Documentation for tambora, a modular N-body Python package for small galactic dynamics tasks"

git branch -M master
git push -u origin master

# Publish from the Actions workflow rather than a branch.
gh api -X POST "repos/$USER/$REPO/pages" \
  -f "build_type=workflow" 2>/dev/null \
  || gh api -X PUT "repos/$USER/$REPO/pages" -f "build_type=workflow"

echo
echo "Pushed. The docs workflow is building now:"
echo "  https://github.com/$USER/$REPO/actions"
echo "It will publish to:"
echo "  https://$USER.github.io/$REPO/"
