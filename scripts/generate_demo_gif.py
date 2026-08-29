"""Script to record and generate the official promptdiff 60fps Terminal Demo GIF using Charmbracelet VHS."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def generate_demo_gif() -> int:
    """Generate high-quality 60fps terminal demo GIF from demo.tape."""
    project_root = Path(__file__).resolve().parent.parent
    tape_file = project_root / "demo.tape"
    assets_dir = project_root / "assets"
    output_gif = assets_dir / "demo.gif"

    assets_dir.mkdir(parents=True, exist_ok=True)

    if not tape_file.is_file():
        print(f"[!] Error: Tape file not found at {tape_file}", file=sys.stderr)
        return 1

    # Check for VHS binary in system PATH
    vhs_path = shutil.which("vhs")
    if not vhs_path:
        print("[!] Charmbracelet VHS is not installed or not in PATH.", file=sys.stderr)
        print("\nTo install VHS:", file=sys.stderr)
        print("  - macOS:   brew install vhs", file=sys.stderr)
        print("  - Windows: winget install charmbracelet.vhs  OR  choco install vhs", file=sys.stderr)
        print("  - Linux:   sudo apt install vhs  OR  go install github.com/charmbracelet/vhs@latest", file=sys.stderr)
        print("\nOnce installed, run:\n  python scripts/generate_demo_gif.py", file=sys.stderr)
        return 1

    print(f"[+] Found VHS executable: {vhs_path}")
    print(f"[+] Recording terminal session from {tape_file} -> {output_gif} ...")

    try:
        result = subprocess.run(
            [vhs_path, str(tape_file)],
            cwd=str(project_root),
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if output_gif.exists():
            size_mb = output_gif.stat().st_size / (1024 * 1024)
            print(f"[✓] Successfully generated demo GIF: {output_gif} ({size_mb:.2f} MB)")
            return 0
        else:
            print(f"[!] Warning: VHS completed but {output_gif} was not created.", file=sys.stderr)
            return 1
    except subprocess.CalledProcessError as e:
        print(f"[!] VHS execution failed: {e.stderr}", file=sys.stderr)
        return e.returncode
    except Exception as e:
        print(f"[!] Unexpected error during demo generation: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(generate_demo_gif())
