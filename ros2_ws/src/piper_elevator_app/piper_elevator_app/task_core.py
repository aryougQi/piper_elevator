"""Pure command parsing helpers for the elevator task manager."""

from dataclasses import dataclass
import shlex


@dataclass(frozen=True)
class TaskCommand:
    """A validated high-level elevator command."""

    action: str
    button: str = ''


PRESS_ALIASES = {'press', 'push', '按压'}
STOP_ALIASES = {'stop', 'cancel', '停止', '取消'}


def parse_task_command(text):
    """Parse ``press 3`` while also accepting a bare button name."""
    value = str(text).strip()
    if not value:
        raise ValueError('command is empty')
    if len(value) > 80 or any(ord(character) < 32 for character in value):
        raise ValueError('command contains invalid characters')

    normalized = value.casefold()
    if normalized in STOP_ALIASES:
        return TaskCommand(action='stop')

    for separator in (':', '='):
        prefix, found, suffix = value.partition(separator)
        if found and prefix.strip().casefold() in PRESS_ALIASES:
            button = suffix.strip()
            if not button:
                raise ValueError('press command has no button')
            return TaskCommand(action='press', button=button)

    try:
        tokens = shlex.split(value)
    except ValueError as error:
        raise ValueError(f'invalid command quoting: {error}') from error
    if not tokens:
        raise ValueError('command is empty')
    if tokens[0].casefold() in PRESS_ALIASES:
        button = ' '.join(tokens[1:]).strip()
        if not button:
            raise ValueError('press command has no button')
        return TaskCommand(action='press', button=button)
    if len(tokens) == 1:
        return TaskCommand(action='press', button=tokens[0])
    raise ValueError('expected `press <button>` or a single button name')
