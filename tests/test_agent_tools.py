import subprocess

import pytest

from dark_factory.agents import tools


def test_read_file_returns_contents(tmp_path):
    (tmp_path / "a.py").write_text("print('hi')")
    assert tools.read_file(tmp_path, "a.py") == "print('hi')"


def test_read_file_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        tools.read_file(tmp_path, "missing.py")


def test_read_file_truncates_long_content(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "MAX_READ_CHARS", 10)
    (tmp_path / "big.txt").write_text("x" * 100)
    result = tools.read_file(tmp_path, "big.txt")
    assert result.startswith("x" * 10)
    assert "truncated" in result


def test_read_file_rejects_path_escaping_root(tmp_path):
    outside = tmp_path.parent / "escaped-secret.txt"
    outside.write_text("secret")
    try:
        with pytest.raises(tools.PathEscapesRootError):
            tools.read_file(tmp_path, "../escaped-secret.txt")
    finally:
        outside.unlink()


def test_write_file_creates_parent_dirs(tmp_path):
    tools.write_file(tmp_path, "nested/dir/hello.txt", "Hello, World!")
    assert (tmp_path / "nested" / "dir" / "hello.txt").read_text() == "Hello, World!"


def test_write_file_rejects_path_escaping_root(tmp_path):
    with pytest.raises(tools.PathEscapesRootError):
        tools.write_file(tmp_path, "../escaped.txt", "pwned")
    assert not (tmp_path.parent / "escaped.txt").exists()


def test_edit_file_replaces_unique_match(tmp_path):
    (tmp_path / "a.py").write_text("def hello():\n    return 'hi'\n")
    tools.edit_file(tmp_path, "a.py", "return 'hi'", "return 'Hello, World!'")
    assert "Hello, World!" in (tmp_path / "a.py").read_text()


def test_edit_file_rejects_no_match(tmp_path):
    (tmp_path / "a.py").write_text("def hello(): pass\n")
    with pytest.raises(ValueError, match="not found"):
        tools.edit_file(tmp_path, "a.py", "does not exist", "x")


def test_edit_file_rejects_ambiguous_match(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\nx = 1\n")
    with pytest.raises(ValueError, match="must be unique"):
        tools.edit_file(tmp_path, "a.py", "x = 1", "x = 2")


def test_edit_file_rejects_path_escaping_root(tmp_path):
    with pytest.raises(tools.PathEscapesRootError):
        tools.edit_file(tmp_path, "../evil.py", "a", "b")


def test_list_files_matches_glob_and_respects_exclude(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.py").write_text("")

    result = tools.list_files(tmp_path, "**/*.py")

    assert "src/main.py" in result
    assert "node_modules" not in result


def test_grep_falls_back_to_plain_scan_without_git(tmp_path):
    (tmp_path / "a.py").write_text("def hello():\n    return 'Hello, World!'\n")
    result = tools.grep(tmp_path, "Hello, World!")
    assert "a.py:2:" in result


def test_grep_uses_git_grep_in_a_real_repo(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "a.py").write_text("def hello():\n    return 'Hello, World!'\n")
    result = tools.grep(tmp_path, "Hello, World!")
    assert "a.py:2:" in result


def test_grep_reports_no_matches(tmp_path):
    (tmp_path / "a.py").write_text("nothing interesting here\n")
    assert tools.grep(tmp_path, "needle") == "(no matches)"


def test_run_command_allows_matching_prefix(tmp_path):
    result = tools.run_command(tmp_path, "echo hi", allowed_prefixes=("echo",))
    assert "hi" in result
    assert "exit code: 0" in result


def test_run_command_rejects_disallowed_command(tmp_path):
    with pytest.raises(PermissionError):
        tools.run_command(tmp_path, "rm -rf /", allowed_prefixes=("pytest",))


def test_run_command_truncates_long_output(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "MAX_COMMAND_OUTPUT_CHARS", 20)
    result = tools.run_command(
        tmp_path, "echo hello-world-this-is-long", allowed_prefixes=("echo",)
    )
    assert "truncated" in result
