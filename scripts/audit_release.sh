#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

status=0

check_pattern() {
  local label="$1"
  local pattern="$2"
  local matches
  matches="$(rg -n -i --hidden \
    --glob '!.git/**' \
    --glob '!scripts/audit_release.sh' \
    --glob '!ANONYMIZATION.md' \
    "$pattern" . || true)"
  if [[ -n "$matches" ]]; then
    echo "[FAIL] $label"
    echo "$matches"
    status=1
  else
    echo "[ OK ] $label"
  fi
}

check_pattern "No absolute private filesystem paths" '(/home/|/nfs/|/users/|/scratch/|/gpfs/)'
check_pattern "No institutional or personal email addresses" '([[:alnum:]._%+-]+@(umich|pes|gmail|yahoo|outlook)\.[[:alpha:]]+)'
check_pattern "No embedded credentials" '(api[_-]?key|secret[_-]?key|access[_-]?token)[[:space:]]*[:=][[:space:]]*[[:alnum:]_./+-]{12,}'
check_pattern "No direct participant identifiers" '(prolific[_-]?(pid|id)|response[_-]?id|worker[_-]?id)'

large_files="$(find . -xdev -type f -size +20M -not -path './.git/*' -print)"
if [[ -n "$large_files" ]]; then
  echo "[FAIL] Files larger than 20 MB"
  echo "$large_files"
  status=1
else
  echo "[ OK ] No files larger than 20 MB"
fi

if find . -xdev -type f \( -name '*.pdf' -o -name '*.xlsx' -o -name '*.numbers' \) -not -path './.git/*' | grep -q .; then
  echo "[FAIL] Review-sensitive document or spreadsheet files found"
  find . -xdev -type f \( -name '*.pdf' -o -name '*.xlsx' -o -name '*.numbers' \) -not -path './.git/*' -print
  status=1
else
  echo "[ OK ] No PDFs or raw spreadsheet workbooks"
fi

exit "$status"

