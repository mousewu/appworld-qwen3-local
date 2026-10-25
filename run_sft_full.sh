#!/usr/bin/env bash
# Default full-parameter SFT now uses Slime. Legacy HF: run_sft_full_hf.sh.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_sft_slime.sh" "$@"
