"""Config loading and auditing for the Session 31 loss kit.

``kit_config`` deep-merges the ``_kit.yaml`` defaults with a per-model file and
enforces the four fail-loud fairness rules; ``audit`` renders the resolved
on/off matrix for review.
"""
