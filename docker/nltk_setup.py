"""Download the NLTK resources used for VADER polarity and named-entity
recognition. Failures are non-fatal: the package falls back to its built-in
lexicon and heuristic NER."""
import os

import nltk

target = os.environ.get("NLTK_DATA", "/usr/share/nltk_data")
for pkg in [
    "vader_lexicon", "punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
    "maxent_ne_chunker", "maxent_ne_chunker_tab", "words",
]:
    try:
        nltk.download(pkg, download_dir=target, quiet=True)
    except Exception as exc:  # noqa: BLE001
        print(f"warning: could not download {pkg}: {exc}")
