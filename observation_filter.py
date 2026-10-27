"""Shared bounded observation policy for RL and evaluation."""
def bound_observation(tokenizer, observation, limit):
    marker = '\n[OUTPUT TRUNCATED: use pagination, filtering or a summary; REPL variables remain available.]'
    # Bound CPU tokenization too, before inspecting huge environment strings.
    shortened = observation[:limit*8]
    encoded = tokenizer(shortened, add_special_tokens=False)['input_ids']
    if len(encoded) <= limit and len(shortened) == len(observation):
        return observation, False
    marker_ids = tokenizer(marker, add_special_tokens=False)['input_ids']
    keep = max(0, limit-len(marker_ids))
    return tokenizer.decode(encoded[:keep], skip_special_tokens=False)+marker, True

