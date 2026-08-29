#!/usr/bin/env bash
# One-time setup: create the repo, push, and turn on GitHub Pages.
# Run this after `gh auth login`.
set -euo pipefail

USER="${1:-sgpfaff}"
REPO="${2:-tambora-docs}"

gh repo create "$USER/$REPO" --public --source=. --remote=origin \
  --description "Documentation for tambora, an N-body code for the modern era"

git branch -M main
git push -u origin main

# Publish from the Actions workflow rather than a branch.
gh api -X POST "repos/$USER/$REPO/pages" \
  -f "build_type=workflow" 2>/dev/null \
  || gh api -X PUT "repos/$USER/$REPO/pages" -f "build_type=workflow"

echo
echo "Pushed. The docs workflow is building now:"
echo "  https://github.com/$USER/$REPO/actions"
echo "It will publish to:"
echo "  https://$USER.github.io/$REPO/"
