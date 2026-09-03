"""CI bind check must fail when 0.0.0.0 is accepted or used as a server bind."""

import ast
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BIND_CHECK = REPO_ROOT / "scripts" / "check_bind_localhost.py"


def test_bind_check_script_names_rejected_wildcard():
    tree = ast.parse(BIND_CHECK.read_text(encoding="utf-8"))
    values = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and node.value == "0.0.0.0"
    ]
    assert values, "policy script must name 0.0.0.0 as the rejected host"


def test_bind_check_exits_zero_on_this_tree():
    result = subprocess.run(
        ["python3", str(BIND_CHECK)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
