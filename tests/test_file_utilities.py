from datetime import datetime
import os
from pathlib import Path

from core_utilities import file_utilities


def _set_file(path: Path, content: str, timestamp: int) -> None:
    path.write_text(content, encoding="utf-8")
    os.utime(path, (timestamp, timestamp))


def test_backup_file_skips_duplicate_content(tmp_path):
    source = tmp_path / "config.ini"
    backup_directory = tmp_path / "backups"

    _set_file(source, "alpha", 1_700_000_000)
    file_utilities.backup_file(
        str(source),
        backup_directory=str(backup_directory),
        number_of_backups=5,
    )

    _set_file(source, "alpha", 1_700_000_060)
    file_utilities.backup_file(
        str(source),
        backup_directory=str(backup_directory),
        number_of_backups=5,
    )

    _set_file(source, "beta", 1_700_000_120)
    file_utilities.backup_file(
        str(source),
        backup_directory=str(backup_directory),
        number_of_backups=5,
    )

    backups = sorted(p.name for p in backup_directory.iterdir())
    assert len(backups) == 2


def test_backup_file_prunes_old_backups(tmp_path):
    source = tmp_path / "notes.txt"
    backup_directory = tmp_path / "backups"

    for offset, content in enumerate(("one", "two", "three")):
        _set_file(source, content, 1_700_000_000 + 60 * offset)
        file_utilities.backup_file(
            str(source),
            backup_directory=str(backup_directory),
            number_of_backups=2,
        )

    backups = sorted(p.name for p in backup_directory.iterdir())
    assert len(backups) == 2
    expected = [
        datetime.fromtimestamp(1_700_000_060).strftime(
            "notes-%Y%m%dT%H%M%S.000.txt"
        ),
        datetime.fromtimestamp(1_700_000_120).strftime(
            "notes-%Y%m%dT%H%M%S.000.txt"
        ),
    ]
    assert backups == expected
