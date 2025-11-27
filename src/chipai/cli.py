import subprocess
import sys
from importlib.resources import files
from pathlib import Path
import os

def main():
    """
    Runs the Streamlit application.
    This function is the entry point for the console script.
    """
    try:
        main_py_path = files('chipai').joinpath('main.py')
        # Assuming the project root is the parent directory of the `chipai` package
        project_root = main_py_path.parent.parent.parent
    except (ModuleNotFoundError, AttributeError):
        print("Error: Could not locate the application's main script.", file=sys.stderr)
        sys.exit(1)

    command = [sys.executable, "-m", "streamlit", "run", str(main_py_path)]

    try:
        # Run streamlit from the project root directory
        subprocess.run(command, check=True, cwd=project_root)
    except FileNotFoundError:
        print("Error: 'streamlit' command not found. Make sure Streamlit is installed and in your PATH.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
