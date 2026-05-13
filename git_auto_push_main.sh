#!/usr/bin/env bash
set -Eeuo pipefail

# Auto commit and push current project to GitHub main branch using a PAT.
# Safer usage: export GITHUB_TOKEN='ghp_xxx' before running this script.
# This script does NOT store your PAT inside .git/config.

REPO_URL="https://github.com/tydeptrai21042004/codat-hcmiu.git"
REPO_PUSH_URL_BASE="github.com/tydeptrai21042004/codat-hcmiu.git"
BRANCH="main"
DEFAULT_USER="tydeptrai21042004"

COMMIT_MSG="${1:-update project}"
GITHUB_USER="${GITHUB_USERNAME:-$DEFAULT_USER}"
TOKEN="${GITHUB_TOKEN:-${GH_TOKEN:-}}"

if ! command -v git >/dev/null 2>&1; then
  echo "[ERROR] git is not installed. Install git first."
  exit 1
fi

if [[ -z "$TOKEN" ]]; then
  read -r -p "GitHub username [$GITHUB_USER]: " INPUT_USER
  if [[ -n "$INPUT_USER" ]]; then
    GITHUB_USER="$INPUT_USER"
  fi

  read -r -s -p "GitHub PAT/token: " TOKEN
  echo

  if [[ -z "$TOKEN" ]]; then
    echo "[ERROR] PAT/token is empty."
    exit 1
  fi
fi

# Initialize git repository if needed.
if [[ ! -d ".git" ]]; then
  echo "[INFO] Initializing git repository..."
  git init
fi

# Optional local identity fallback. You can override before running:
# git config user.name "Your Name"
# git config user.email "your-email@example.com"
if ! git config user.name >/dev/null; then
  git config user.name "$GITHUB_USER"
fi
if ! git config user.email >/dev/null; then
  git config user.email "$GITHUB_USER@users.noreply.github.com"
fi

# Ensure main branch.
echo "[INFO] Switching branch to $BRANCH..."
git branch -M "$BRANCH"

# Add/update clean remote URL without token.
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REPO_URL"
else
  git remote add origin "$REPO_URL"
fi

# Stage all current code, not only README.md.
echo "[INFO] Staging files..."
git add -A

# Commit only if there are changes.
if git diff --cached --quiet; then
  echo "[INFO] No new changes to commit."
else
  echo "[INFO] Creating commit: $COMMIT_MSG"
  git commit -m "$COMMIT_MSG"
fi

# Push using token only for this command; token is not saved to remote config.
AUTH_REMOTE="https://${GITHUB_USER}:${TOKEN}@${REPO_PUSH_URL_BASE}"

echo "[INFO] Pushing to origin/$BRANCH..."
git push -u "$AUTH_REMOTE" "$BRANCH"

echo "[DONE] Project pushed to $REPO_URL on branch $BRANCH."
