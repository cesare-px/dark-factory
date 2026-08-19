from pathlib import Path

from dark_factory.config.model import TestHarnessSettings
from dark_factory.harness import ScriptedHarness, SubprocessHarness


def test_scripted_harness_fails_then_passes():
    harness = ScriptedHarness(passes_on_attempt=3)
    assert harness.run(iteration=1, workdir=Path(".")).passed is False
    assert harness.run(iteration=2, workdir=Path(".")).passed is False
    assert harness.run(iteration=3, workdir=Path(".")).passed is True


def test_subprocess_harness_runs_real_command(tmp_path: Path):
    cfg = TestHarnessSettings(command="python3 -c \"print('ok')\"")
    harness = SubprocessHarness(cfg)
    result = harness.run(iteration=1, workdir=tmp_path)
    assert result.passed is True
    assert "ok" in result.stdout


def test_subprocess_harness_reports_failure_exit_code(tmp_path: Path):
    cfg = TestHarnessSettings(command='python3 -c "import sys; sys.exit(1)"')
    harness = SubprocessHarness(cfg)
    result = harness.run(iteration=1, workdir=tmp_path)
    assert result.passed is False
    assert result.exit_code == 1


def test_subprocess_harness_truncates_long_output(tmp_path: Path):
    cfg = TestHarnessSettings(command="python3 -c \"print('x' * 100)\"", max_output_chars=10)
    harness = SubprocessHarness(cfg)
    result = harness.run(iteration=1, workdir=tmp_path)
    assert len(result.stdout) <= 10
    assert result.truncated is True


def test_subprocess_harness_missing_command_fails_gracefully(tmp_path: Path):
    cfg = TestHarnessSettings(command="./factory-test.sh")  # does not exist in tmp_path
    harness = SubprocessHarness(cfg)
    result = harness.run(iteration=1, workdir=tmp_path)
    assert result.passed is False
    assert result.exit_code == 127
    assert "could not run" in result.stderr


def test_subprocess_harness_runs_setup_command_first(tmp_path: Path):
    marker = tmp_path / "marker.txt"
    cfg = TestHarnessSettings(
        command=f"python3 -c \"import pathlib; print(pathlib.Path('{marker}').exists())\"",
        setup_command=f"python3 -c \"open('{marker}', 'w').close()\"",
    )
    harness = SubprocessHarness(cfg)
    result = harness.run(iteration=1, workdir=tmp_path)
    assert "True" in result.stdout
