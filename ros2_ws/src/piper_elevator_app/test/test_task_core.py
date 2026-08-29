"""Unit tests for high-level task commands."""

import pytest

from piper_elevator_app.task_core import parse_task_command


@pytest.mark.parametrize(
    'text,button',
    [
        ('press 3', '3'),
        ('push up', 'up'),
        ('按压 down', 'down'),
        ('press:open', 'open'),
        ('press=close', 'close'),
        ('B1', 'B1'),
    ],
)
def test_parse_press_command(text, button):
    command = parse_task_command(text)

    assert command.action == 'press'
    assert command.button == button


@pytest.mark.parametrize('text', ['stop', 'cancel', '停止', '取消'])
def test_parse_stop_command(text):
    command = parse_task_command(text)

    assert command.action == 'stop'
    assert command.button == ''


@pytest.mark.parametrize('text', ['', 'press', 'press:', 'unknown two words'])
def test_reject_incomplete_or_ambiguous_command(text):
    with pytest.raises(ValueError):
        parse_task_command(text)
