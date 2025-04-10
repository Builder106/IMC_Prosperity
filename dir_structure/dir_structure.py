import os
from pathlib import Path
import sys

def list_directory_structure(startpath):
    """
    Prints the directory structure starting from the given path
    in a tree-like format.
    """
    try:
        # Convert the input string path to a Path object
        root_dir = Path(startpath)

        # Basic validation
        if not root_dir.exists():
            print(f"Error: Path does not exist: {startpath}", file=sys.stderr)
            return
        if not root_dir.is_dir():
            print(f"Error: Path is not a directory: {startpath}", file=sys.stderr)
            return

        # Print the root directory name itself
        print(f"{root_dir.name}/")

        # Start the recursive listing process
        _list_dir_recursive(root_dir)

    except PermissionError:
        # Handle cases where the script might not have read permissions
        # for the root directory itself (less common)
        print(f"Error: Permission denied for directory: {startpath}", file=sys.stderr)
    except Exception as e:
        # Catch any other unexpected errors during initial setup
        print(f"An unexpected error occurred: {e}", file=sys.stderr)


def _list_dir_recursive(directory_path, prefix=""):
    """
    Internal helper function to recursively list directory contents.

    Args:
        directory_path (Path): The Path object of the current directory to list.
        prefix (str): The string prefix (containing spaces and pipe characters)
                      to prepend to the current level's items for formatting.
    """
    try:
        # Get directory contents, convert iterator to list to easily check the last item
        # Sort entries for consistent output (folders might appear before/after files depending on OS default)
        # Here, we sort alphabetically; you could sort folders first if desired.
        entries = sorted(list(directory_path.iterdir()), key=lambda x: x.name)
        # entries = sorted(list(directory_path.iterdir()), key=lambda x: (not x.is_dir(), x.name)) # Sort folders first

        # Define the connectors for tree branches
        pointers = ["├── "] * (len(entries) - 1) + ["└── "]

        for i, entry in enumerate(entries):
            pointer = pointers[i]
            entry_name = entry.name
            is_dir = entry.is_dir()

            # Print the current entry (file or directory)
            print(f"{prefix}{pointer}{entry_name}{'/' if is_dir else ''}")

            if is_dir:
                # For directories, make a recursive call
                # Determine the extension for the prefix in the next level:
                # If this was the last item, use spaces; otherwise, use a pipe.
                extension = "│   " if i < len(entries) - 1 else "    "
                _list_dir_recursive(entry, prefix + extension)

    except PermissionError:
        # Handle permission errors for subdirectories gracefully
        print(f"{prefix}└── [Permission Denied: {directory_path.name}/]", file=sys.stderr)
    except Exception as e:
        # Catch other errors during listing (e.g., broken symlinks if not handled)
        print(f"{prefix}└── [Error listing {directory_path.name}/: {e}]", file=sys.stderr)


# --- Main execution part ---
if __name__ == "__main__":
    folder_path_input = input("Enter the path to the folder: ")
    # Basic input cleaning (remove potential surrounding quotes)
    folder_path_input = folder_path_input.strip('\'"')
    list_directory_structure(folder_path_input)