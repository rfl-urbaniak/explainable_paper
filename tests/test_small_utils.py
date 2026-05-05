from pathlib import Path

from pci.tools.find_root import find_repo_root


def test_find_root():
    test_path = Path(__file__).parent.parent
    root_path = find_repo_root(test_path)

    assert test_path == root_path
