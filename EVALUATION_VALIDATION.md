# Validation and publication scope

- Base repository: krith31/Telecom-Hybrid-RAG-Assistant.
- Existing files are unchanged; new result_accuracy.py, result_faithfulness.py and capture.py avoid replacing the original evaluation modules.
- The previously delivered overlay workflow passed 34 offline tests on Python 3.12.14.
- This additive version adapts module names and imports to preserve all existing files. It includes offline regression tests, but it has not been re-executed because the local executor is unavailable.
- GitHub tree comparison verifies that publication adds files only.
- Live model calls and retrieval against the user's populated corpus have not been run. No production quality score is claimed.
- Run the commands in EVALUATION_README.md in the configured RAG environment.
