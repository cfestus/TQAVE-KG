# Makes tests/ an importable package so `python -m unittest discover -v`,
# run from the repository root, recurses into this directory instead of
# silently skipping it (Python's unittest discovery, pre-3.11, only walks
# into subdirectories that are regular packages).
