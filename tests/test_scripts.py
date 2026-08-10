"""Behavioral checks for local PowerShell safety wrappers."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = (
    Path(os.environ.get("WINDIR", r"C:\Windows"))
    / "System32/WindowsPowerShell/v1.0/powershell.exe"
)


def _run(script: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(POWERSHELL),
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / script),
            *arguments,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )


@pytest.mark.skipif(not POWERSHELL.exists(), reason="Windows PowerShell is unavailable")
def test_readiness_wrapper_rejects_unknown_broker_at_parameter_binding() -> None:
    """An arbitrary broker string must never become a command argument."""
    result = _run("check-readiness.ps1", "-Broker", "attacker;whoami")

    assert result.returncode != 0
    assert "whoami" not in result.stdout
    assert "api_key" not in (result.stdout + result.stderr).lower()


@pytest.mark.skipif(not POWERSHELL.exists(), reason="Windows PowerShell is unavailable")
def test_export_wrapper_handles_quoted_absolute_path_without_injection(tmp_path: Path) -> None:
    """A destination containing spaces and punctuation must remain one inert path value."""
    destination = tmp_path / "safe folder" / "status & still-safe.json"
    destination.parent.mkdir()
    result = _run(
        "export-dashboard-status.ps1",
        "-Broker",
        "alpaca",
        "-Path",
        str(destination),
    )

    assert result.returncode == 0, result.stderr
    assert destination.exists()
    assert not (tmp_path / "still-safe.json").exists()
    dashboard = json.loads(destination.read_text(encoding="utf-8"))
    assert dashboard["brokers"] == [
        {
            "missing_gates": sorted(
                (
                    "ALPACA_ACCOUNT_ACTIVE",
                    "ALPACA_ACCOUNT_ID_MATCHED",
                    "ALPACA_ACCOUNT_ID_PRESENT",
                    "ALPACA_ACCOUNT_UNBLOCKED",
                    "ALPACA_LIVE_ENDPOINT",
                    "ALPACA_LIVE_TRADING_ENABLED",
                    "ALPACA_LOCAL_CREDENTIALS_PRESENT",
                    "ALPACA_REAL_API_ENABLED",
                    "ALPACA_SUFFICIENT_BUYING_POWER",
                    "MARKET_SENTINEL_MODE",
                )
            ),
            "name": "alpaca",
            "ready": False,
        }
    ]


@pytest.mark.skipif(not POWERSHELL.exists(), reason="Windows PowerShell is unavailable")
def test_wrapper_parameter_surface_has_no_secret_bearing_parameters() -> None:
    """PowerShell command discovery must expose only the documented safe parameters."""
    excluded = ",".join(
        f"'{name}'"
        for name in (
            "Verbose",
            "Debug",
            "ErrorAction",
            "WarningAction",
            "InformationAction",
            "ProgressAction",
            "ErrorVariable",
            "WarningVariable",
            "InformationVariable",
            "OutVariable",
            "OutBuffer",
            "PipelineVariable",
        )
    )
    command = (
        "$commands = @('check-readiness.ps1','export-dashboard-status.ps1'); "
        "foreach ($name in $commands) { "
        "$c = Get-Command (Join-Path 'scripts' $name); "
        f"$c.Parameters.Keys | Where-Object {{ $_ -notin @({excluded}) }} | Sort-Object | "
        'ForEach-Object { "$name`:$($_)" } }'
    )
    result = subprocess.run(
        [
            str(POWERSHELL),
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert set(result.stdout.splitlines()) == {
        "check-readiness.ps1:Broker",
        "export-dashboard-status.ps1:Broker",
        "export-dashboard-status.ps1:Path",
    }
