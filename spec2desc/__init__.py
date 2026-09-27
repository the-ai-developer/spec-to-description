"""Spec-to-description: the seq2seq side of the catalogue work.

Three pieces, all readable enough to check by hand:

- ``linearise``   spec fields -> one input string (the seq2seq contract)
- ``attention``   multi-head attention, encoder/decoder blocks, causal mask
- ``beam_search`` beam search in plain Python, no ML libraries

Production weights come from ``train/train_t5_description.py`` (HuggingFace
T5). Everything here exists so the maths can be audited against it.
"""

from .linearise import linearise_spec

__all__ = ["linearise_spec"]
