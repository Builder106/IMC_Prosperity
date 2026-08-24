from src.utils import dir_structure


class _RootThatRaises:
    def __init__(self, error):
        self.error = error

    def exists(self):
        raise self.error


def test_list_directory_structure_handles_root_permission_error(monkeypatch, capsys):
    monkeypatch.setattr(
        dir_structure,
        "Path",
        lambda _: _RootThatRaises(PermissionError()),
    )

    dir_structure.list_directory_structure("restricted")

    assert "Error: Permission denied for directory: restricted" in capsys.readouterr().err


def test_list_directory_structure_handles_root_os_error(monkeypatch, capsys):
    monkeypatch.setattr(
        dir_structure,
        "Path",
        lambda _: _RootThatRaises(OSError("filesystem unavailable")),
    )

    dir_structure.list_directory_structure("broken")

    assert "An unexpected error occurred: filesystem unavailable" in capsys.readouterr().err


class _DirectoryThatRaises:
    name = "nested"

    def __init__(self, error):
        self.error = error

    def iterdir(self):
        raise self.error


def test_recursive_listing_handles_permission_error(capsys):
    dir_structure._list_dir_recursive(_DirectoryThatRaises(PermissionError()), "│   ")

    assert "│   └── [Permission Denied: nested/]" in capsys.readouterr().err


def test_recursive_listing_handles_os_error(capsys):
    dir_structure._list_dir_recursive(
        _DirectoryThatRaises(OSError("directory vanished")),
        "    ",
    )

    assert "    └── [Error listing nested/: directory vanished]" in capsys.readouterr().err


def test_list_directory_structure_prints_sorted_tree(tmp_path, capsys):
    root = tmp_path / "root"
    (root / "nested").mkdir(parents=True)
    (root / "nested" / "child.txt").write_text("data", encoding="utf-8")
    (root / "top.txt").write_text("data", encoding="utf-8")

    dir_structure.list_directory_structure(root)

    output = capsys.readouterr().out
    assert output.splitlines() == [
        "root/",
        "├── nested/",
        "│   └── child.txt",
        "└── top.txt",
    ]


def test_list_directory_structure_reports_invalid_paths(tmp_path, capsys):
    missing = tmp_path / "missing"
    file_path = tmp_path / "file.txt"
    file_path.write_text("data", encoding="utf-8")

    dir_structure.list_directory_structure(missing)
    assert f"Error: Path does not exist: {missing}" in capsys.readouterr().err

    dir_structure.list_directory_structure(file_path)
    assert f"Error: Path is not a directory: {file_path}" in capsys.readouterr().err


def test_list_directory_structure_handles_empty_directory(tmp_path, capsys):
    root = tmp_path / "empty"
    root.mkdir()

    dir_structure.list_directory_structure(root)

    assert capsys.readouterr().out == "empty/\n"
