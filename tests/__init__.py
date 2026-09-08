"""Test package.

Fake credentials are set here -- the earliest import hook pytest gives us -- so
that config.py can be imported in CI without real tokens. No test ever uses them.
"""

import os

for _name in ("PERSONAL_ACCESS_TOKEN_CLAUDE", "API_TOKEN_TODOIST", "API_TOKEN_TELEGRAM"):
    os.environ.setdefault(_name, "test-token")
