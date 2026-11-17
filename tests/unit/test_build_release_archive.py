"""The deploy archive must carry modelzoo at the pinned commit, and refuse anything else.

``platform/ci/build_release_archive.sh`` replaced a bare ``git archive``: the node used to
fill every release's modelzoo from a snapshot seeded once and never refreshed, so serving
ran months-old model code. These tests run the real script against throwaway git repos.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = ROOT / "platform" / "ci" / "build_release_archive.sh"

pytestmark = pytest.mark.skipif(
    not (shutil.which("bash") and shutil.which("git") and shutil.which("tar")),
    reason="needs bash, git and tar",
)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repo(path: Path, files: dict[str, str]) -> str:
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    for name, text in files.items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_text(text)
    _git(path, "add", "-A")
    _git(path, "commit", "-qm", "c")
    return _git(path, "rev-parse", "HEAD")


@pytest.fixture
def repos(tmp_path):
    zoo = tmp_path / "zoo"
    old = _repo(
        zoo,
        {"seanergys_modelzoo/__init__.py": "OLD = True\n", "data/123/big.parquet": "x" * 64},
    )
    (zoo / "seanergys_modelzoo" / "__init__.py").write_text("NEW = True\n")
    _git(zoo, "commit", "-qam", "new")
    new = _git(zoo, "rev-parse", "HEAD")

    def platform(ref: str) -> tuple[Path, str]:
        app = tmp_path / f"app-{ref[:8]}"
        sha = _repo(app, {"Makefile": f"MODELZOO_REF    ?= {ref}\n", "README": "x\n"})
        return app, sha

    return zoo, old, new, platform


def _build(app: Path, sha: str, zoo: Path, out: Path):
    return subprocess.run(
        ["bash", str(_SCRIPT), str(out), sha],
        cwd=app,
        env={"PATH": "/usr/bin:/bin", "MODELZOO_SOURCE": str(zoo), "HOME": str(app)},
        capture_output=True,
        text=True,
    )


def test_the_archive_carries_modelzoo_at_the_pinned_commit_not_the_newest(repos, tmp_path):
    zoo, old, _new, platform = repos
    app, sha = platform(old)
    out = tmp_path / "release.tar.gz"

    result = _build(app, sha, zoo, out)
    assert result.returncode == 0, result.stderr

    with tarfile.open(out) as tar:
        names = set(tar.getnames())
        init = tar.extractfile("modelzoo/seanergys_modelzoo/__init__.py").read().decode()
        marker = tar.extractfile("modelzoo/.examlops-modelzoo-ref").read().decode().strip()
    assert "README" in names and "Makefile" in names, "the platform tree itself is missing"
    assert init == "OLD = True\n"
    assert marker == old
    assert not any(n.startswith("modelzoo/data") for n in names), "dataset cache was shipped"


@pytest.mark.parametrize("bad", ["main", "ccf98b0", ""])
def test_a_ref_that_is_not_a_full_commit_id_is_refused(repos, tmp_path, bad):
    zoo, _old, _new, platform = repos
    app, sha = platform(bad)

    result = _build(app, sha, zoo, tmp_path / "r.tar.gz")
    assert result.returncode != 0
    assert "full 40-character commit id" in result.stderr


def test_a_pin_modelzoo_does_not_have_fails_the_build(repos, tmp_path):
    zoo, _old, _new, platform = repos
    app, sha = platform("0" * 40)

    result = _build(app, sha, zoo, tmp_path / "r.tar.gz")
    assert result.returncode != 0
    assert "has no commit" in result.stderr
    assert not (tmp_path / "r.tar.gz").exists()


def test_the_real_pin_is_a_full_commit_id():
    ref = re.search(r"^MODELZOO_REF\s*\?=\s*(\S+)", (ROOT / "Makefile").read_text(), re.M)
    assert ref and re.fullmatch(r"[0-9a-f]{40}", ref.group(1)), ref


def test_the_deploy_job_builds_its_archive_with_the_script():
    ci = (ROOT / ".gitlab-ci.yml").read_text()
    assert "platform/ci/build_release_archive.sh" in ci
    assert 'git archive --format=tar.gz -o "$ARCHIVE"' not in ci


def test_without_any_token_the_error_says_what_is_missing(repos, tmp_path):
    zoo, old, _new, platform = repos
    app, sha = platform(old)
    result = subprocess.run(
        ["bash", str(_SCRIPT), str(tmp_path / "r.tar.gz"), sha],
        cwd=app,
        env={"PATH": "/usr/bin:/bin", "HOME": str(app), "CI_SERVER_HOST": "gitlab.invalid"},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "neither GITLAB_TOKEN nor CI_JOB_TOKEN" in result.stderr
