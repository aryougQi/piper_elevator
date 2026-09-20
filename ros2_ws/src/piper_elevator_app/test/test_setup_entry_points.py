"""Packaging contracts for application executables used by launch files."""

import ast
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_simulation_servo_adapter_is_installed():
    tree = ast.parse((PACKAGE_ROOT / 'setup.py').read_text(encoding='utf-8'))
    strings = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert any(
        'simulation_servo_adapter = ' in value
        and 'piper_elevator_app.simulation_servo_adapter:main' in value
        for value in strings
    )
