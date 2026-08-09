from typer.testing import CliRunner

from market_sentinel import __version__
from market_sentinel.cli import app


def test_package_exposes_version() -> None:
    assert __version__ == "0.1.0"


def test_status_command_is_safe_by_default() -> None:
    result = CliRunner().invoke(app, ["status"])
    assert result.exit_code == 0
    assert "mode=research" in result.stdout
    assert "live_ready=false" in result.stdout
