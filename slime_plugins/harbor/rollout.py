"""Expose AppWorld's native SGLang scheduler at Terminal-RSI's validated path.

Terminal-RSI deliberately restricts TB-GAE to a reviewed rollout entry point.
This project preserves that path while delegating scheduling to Slime's own
SGLang rollout.  The per-sample environment interaction remains the explicitly
configured ``appworld_rpr.generate`` hook.
"""
from slime.rollout.sglang_rollout import generate_rollout as _generate_rollout


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    return _generate_rollout(args, rollout_id, data_source, evaluation=evaluation)
