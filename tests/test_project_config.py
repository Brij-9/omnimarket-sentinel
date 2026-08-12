"""Executable repository, documentation, packaging, and GitHub CI boundaries."""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from packaging.markers import default_environment
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
REQUIRED_JOBS = {"lint-type", "unit", "fixture-integration", "security", "package"}
LIVE_FALSE_KEYS = {
    "ALPACA_LIVE_TRADING_ENABLED",
    "ALPACA_REAL_API_ENABLED",
    "INDIA_LIVE_TRADING_ENABLED",
    "GROWW_REAL_API_ENABLED",
    "CCXT_LIVE_TRADING_ENABLED",
    "CCXT_REAL_API_ENABLED",
}


def _workflow() -> dict[str, object]:
    loaded = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert type(loaded) is dict
    return loaded


def _walk(value: object) -> list[object]:
    found = [value]
    if isinstance(value, Mapping):
        for key, item in value.items():
            found.extend(_walk(key))
            found.extend(_walk(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            found.extend(_walk(item))
    return found


def test_ci_contains_required_offline_quality_jobs_and_python_312() -> None:
    """Dropping a quality gate or changing runtimes would leave acceptance evidence unverified."""
    workflow = _workflow()
    jobs = workflow["jobs"]
    assert isinstance(jobs, Mapping)
    assert set(jobs) >= REQUIRED_JOBS
    flattened = "\n".join(str(value) for value in _walk(workflow))
    assert "3.12" in flattened
    assert "ruff check ." in flattened
    assert "mypy --strict src/market_sentinel" in flattened
    assert 'not integration and not e2e' in flattened
    assert 'integration or e2e' in flattened
    assert "pip-audit" in flattened
    assert "python -m build" in flattened
    assert "market_sentinel.cli status" in flattened
    assert "--require-hashes" in flattened
    assert "python scripts/scan_repository_secrets.py" in flattened


def test_ci_forces_live_flags_false_and_defines_no_credentials() -> None:
    """CI must be structurally incapable of becoming a credentialed real-money environment."""
    workflow = _workflow()
    environment = workflow.get("env")
    assert isinstance(environment, Mapping)
    assert set(environment) >= LIVE_FALSE_KEYS
    assert all(str(environment[key]).lower() == "false" for key in LIVE_FALSE_KEYS)
    forbidden = re.compile(
        r"(?:API_KEY|SECRET|TOKEN|PASSWORD|PASSPHRASE|CREDENTIAL|ACCOUNT_ID|LLM_PROVIDER)",
        re.IGNORECASE,
    )
    all_mappings = [value for value in _walk(workflow) if isinstance(value, Mapping)]
    for mapping in all_mappings:
        if "env" not in mapping or not isinstance(mapping["env"], Mapping):
            continue
        assert not any(forbidden.search(str(key)) for key in mapping["env"])


def test_ci_actions_are_immutable_and_permissions_are_read_only() -> None:
    """A floating action or write permission would weaken supply-chain containment."""
    workflow = _workflow()
    uses = [str(value) for value in _walk(workflow) if isinstance(value, str) and "@" in value]
    action_uses = [value for value in uses if value.startswith("actions/")]
    assert action_uses
    assert all(re.fullmatch(r"actions/[a-z0-9-]+@[0-9a-f]{40}", value) for value in action_uses)
    assert workflow["permissions"] == {"contents": "read"}


def test_repository_secret_scanner_fails_closed_without_printing_secret_contents(
    tmp_path: Path,
) -> None:
    """The CI scanner must execute against controlled inputs and redact every match."""
    scanner = ROOT / "scripts" / "scan_repository_secrets.py"
    clean = tmp_path / "clean.txt"
    false_positive = tmp_path / "false-positive.txt"
    suspect = tmp_path / "suspect.txt"
    clean.write_text("sanitized fixture text", encoding="utf-8")
    false_positive.write_text(
        "task-14-live-test-safety-key-material",
        encoding="utf-8",
    )
    task16_paths = (
        scanner,
        ROOT / ".github" / "workflows" / "ci.yml",
        ROOT / "README.md",
        ROOT / "SECURITY.md",
        ROOT / "docs" / "operations" / "live-readiness.md",
        ROOT / "docs" / "operations" / "strategy-validation.md",
        ROOT / "docs" / "verification" / "2026-08-09-completion-audit.md",
        ROOT
        / "docs"
        / "superpowers"
        / "specs"
        / "2026-08-09-omnimarket-sentinel-design.md",
        ROOT / "src" / "market_sentinel" / "operations" / "fixture_pipeline.py",
        ROOT
        / "src"
        / "market_sentinel"
        / "operations"
        / "fixtures"
        / "e2e_markets.json",
        ROOT / "tests" / "e2e" / "test_cli_fixture_workflows.py",
        ROOT / "tests" / "e2e" / "test_india_pipeline.py",
        ROOT / "tests" / "e2e" / "test_us_pipeline.py",
        ROOT / "tests" / "e2e" / "test_crypto_pipeline.py",
        ROOT / "tests" / "e2e" / "test_live_lock.py",
        ROOT / "tests" / "fixtures" / "e2e_markets.json",
        ROOT / "tests" / "test_project_config.py",
    )
    secret = "AKIA" + "A" * 16
    suspect.write_text(f"prefix {secret} suffix", encoding="utf-8")

    clean_result = subprocess.run(
        (sys.executable, str(scanner), str(clean), str(false_positive)),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    suspect_result = subprocess.run(
        (sys.executable, str(scanner), str(suspect)),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    tracked_result = subprocess.run(
        (sys.executable, str(scanner)),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    task16_result = subprocess.run(
        (sys.executable, str(scanner), *(str(path) for path in task16_paths)),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert clean_result.returncode == 0
    assert tracked_result.returncode == 0
    assert task16_result.returncode == 0
    assert suspect_result.returncode == 1
    assert "secret-pattern matches detected: 1" in suspect_result.stderr
    assert secret not in suspect_result.stdout
    assert secret not in suspect_result.stderr


def test_lock_boundary_keeps_exact_tauric_commit_without_fake_hash() -> None:
    """Inventing a VCS hash or silently dropping the pin would make installs unreproducible."""
    commit = "a33fd4c0f134485a43553a2c23a63cb14adbd88f"
    lock = (ROOT / "requirements.lock").read_text(encoding="utf-8")
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    workflow_text = WORKFLOW.read_text(encoding="utf-8")
    exact = f"tradingagents @ git+https://github.com/TauricResearch/TradingAgents.git@{commit}"
    assert lock.count(exact) == 1
    assert project.count(exact) == 1
    assert exact in workflow_text
    assert not re.search(rf"{re.escape(exact)}\s*\\\s*--hash", lock)
    assert "requirements-hashed.lock" in workflow_text


def test_lock_preserves_ccxt_event_loop_dependencies_across_platforms() -> None:
    """The checked-in lock must resolve CCXT's event loop dependency on CI and Windows."""
    lines = (ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines()

    def requirement_block(name: str) -> list[str]:
        start = next(
            (index for index, line in enumerate(lines) if line.startswith(f"{name}==")),
            None,
        )
        assert start is not None, f"{name} is missing from requirements.lock"
        end = next(
            (
                index
                for index in range(start + 1, len(lines))
                if lines[index]
                and not lines[index].startswith((" ", "#"))
            ),
            len(lines),
        )
        return lines[start:end]

    uvloop = requirement_block("uvloop")
    winloop = requirement_block("winloop")
    uvloop_requirement = Requirement(uvloop[0].removesuffix("\\").strip())
    winloop_requirement = Requirement(winloop[0].removesuffix("\\").strip())
    assert uvloop_requirement.marker is not None
    assert winloop_requirement.marker is not None

    linux = default_environment()
    linux.update(
        platform_system="Linux",
        implementation_name="cpython",
        python_version="3.12",
        platform_machine="x86_64",
    )
    windows = dict(linux)
    windows.update(platform_system="Windows", platform_machine="AMD64")

    assert uvloop_requirement.marker.evaluate(linux)
    assert not uvloop_requirement.marker.evaluate(windows)
    assert not winloop_requirement.marker.evaluate(linux)
    assert winloop_requirement.marker.evaluate(windows)
    assert any(
        "--hash=sha256:7b5b1ac819a3f946d3b2ee07f09149578ae76066d70b44df3fa990add49a82e4"
        in line
        for line in uvloop
    )


def test_package_includes_cli_and_powershell_wrappers() -> None:
    """A built distribution without its CLI or readiness wrappers is not operable."""
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'market-sentinel = "market_sentinel.cli:app"' in project
    assert '"market_sentinel.operations" = ["fixtures/*.json"]' in project
    assert '"scripts/check-readiness.ps1"' in project
    assert '"scripts/export-dashboard-status.ps1"' in project
    assert (ROOT / "scripts" / "check-readiness.ps1").is_file()
    assert (ROOT / "scripts" / "export-dashboard-status.ps1").is_file()
    assert (
        ROOT / "src" / "market_sentinel" / "operations" / "fixtures" / "e2e_markets.json"
    ).read_bytes() == (ROOT / "tests" / "fixtures" / "e2e_markets.json").read_bytes()


def test_package_job_executes_all_sanitized_workflows_from_installed_wheel() -> None:
    """Source-tree tests cannot prove the built wheel includes its fixture data."""
    workflow = WORKFLOW.read_text(encoding="utf-8")
    required = (
        "market_sentinel.cli research --instrument RELIANCE@groww "
        "--as-of 2026-08-10T03:48:30+00:00",
        "market_sentinel.cli backtest --instrument AAPL@alpaca "
        "--start 2026-08-10T13:30:00+00:00 --end 2026-08-10T13:33:30+00:00",
        "market_sentinel.cli paper-run --instrument BTC-USDT@ccxt-spot "
        "--as-of 2026-08-10T00:20:30+00:00",
    )
    assert all(command in workflow for command in required)


def test_required_operator_documents_exist_and_reject_profit_promises() -> None:
    """Operators need explicit limits and must never receive a return guarantee."""
    paths = (
        ROOT / "README.md",
        ROOT / "SECURITY.md",
        ROOT / "docs" / "operations" / "live-readiness.md",
        ROOT / "docs" / "operations" / "strategy-validation.md",
        ROOT / "docs" / "verification" / "2026-08-09-completion-audit.md",
    )
    texts = []
    for path in paths:
        assert path.is_file()
        text = path.read_text(encoding="utf-8")
        assert text.strip()
        texts.append(text)
    combined = "\n".join(texts).lower()
    assert "100,000x" in combined
    assert "extraordinarily improbable" in combined
    assert "usd 10 can be lost in full" in combined
    assert "other markets require explicit" in combined
    assert "i_confirm_real_money_order" in combined
    readme = texts[0].lower()
    security = texts[1].lower()
    assert "generated with python 3.13" in readme
    assert "python 3.12 lock installation was verified" in readme
    assert "security tab" in security
    assert "report a vulnerability" in security
    assert "private contact method" in security
    assert "repository owner's github profile" in security
    forbidden_promises = (
        "guaranteed profit",
        "guaranteed return",
        "will earn $1,000,000",
        "ensures profit",
    )
    assert all(phrase not in combined for phrase in forbidden_promises)


def test_completion_audit_has_one_evidence_row_per_acceptance_criterion() -> None:
    """A completed CI criterion must cite the exact successful public run."""
    text = (
        ROOT / "docs" / "verification" / "2026-08-09-completion-audit.md"
    ).read_text(encoding="utf-8")
    rows = [line for line in text.splitlines() if re.match(r"\|\s*(?:10|[1-9])\s*\|", line)]
    assert len(rows) == 10
    statuses: dict[int, str] = {}
    evidence_rows: dict[int, str] = {}
    for row in rows:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        statuses[int(cells[0])] = cells[-1]
        evidence_rows[int(cells[0])] = row
        assert chr(96) in row
    assert all(status == "Complete" for status in statuses.values())
    assert (
        "https://github.com/Brij-9/omnimarket-sentinel/actions/runs/31631595574"
        in evidence_rows[9]
    )
    assert "ea1765be95fdd9a7254e9c6c98f7f4893b5c2d35" in evidence_rows[9]
