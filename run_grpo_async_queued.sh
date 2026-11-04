#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export NUM_GPUS=4 ACTOR_GPUS=2 TP_SIZE=2 ROLLOUT_TP=1 ASYNC_IMPLEMENTATION=project
exec bash run_grpo_async.sh \
  --env-execute-budget-seconds "${ENV_EXECUTE_BUDGET_SECONDS:-300}" \
  --async-group-capacity "${ASYNC_GROUP_CAPACITY:-16}" \
  --async-inflight-groups "${ASYNC_INFLIGHT_GROUPS:-8}" \
  --async-max-policy-lag "${ASYNC_MAX_POLICY_LAG:-2}" \
  --async-group-retries "${ASYNC_GROUP_RETRIES:-3}" "$@"
