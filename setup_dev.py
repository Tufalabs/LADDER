#!/usr/bin/env python3
"""
Simple development setup script for LADDER.
Replaces the complexity of Makefiles with a straightforward Python approach.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str = "") -> bool:
    """Run a command and return True if successful."""
    if description:
        print(f"→ {description}")
    
    try:
        result = subprocess.run(cmd.split(), check=True, capture_output=True, text=True)
        if result.stdout.strip():
            print(f"  {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Error: {e}")
        if e.stderr:
            print(f"  {e.stderr.strip()}")
        return False
    except FileNotFoundError:
        print(f"  ❌ Command not found: {cmd.split()[0]}")
        return False


def main():
    """Main setup function."""
    print("🪜 LADDER Development Setup")
    print("=" * 30)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
    else:
        print("Available commands:")
        print("  install     - Install package in development mode")
        print("  test        - Run tests")
        print("  format      - Format code with black and isort")
        print("  lint        - Run linting checks")
        print("  type-check  - Run type checking")
        print("  clean       - Clean build artifacts")
        print("  all         - Run format, lint, type-check, and test")
        return
    
    success = True
    
    if command == "install":
        # Try uv first, fall back to pip
        uv_success = run_command("uv sync --dev", "Installing with uv")
        if uv_success:
            print("✅ Installation complete with uv")
            success = True
        else:
            print("⚠️  uv not found, falling back to pip")
            success = run_command("pip install -e .[dev]", "Installing with pip")
            if success:
                print("✅ Installation complete with pip")
    
    elif command == "test":
        success = run_command("pytest", "Running tests")
        if success:
            print("✅ All tests passed")
    
    elif command == "format":
        print("Formatting code...")
        success1 = run_command("black ladder/ tests/", "Running black")
        success2 = run_command("isort ladder/ tests/", "Running isort")
        success = success1 and success2
        if success:
            print("✅ Code formatted")
    
    elif command == "lint":
        success = run_command("flake8 ladder/ tests/", "Running flake8")
        if success:
            print("✅ Linting passed")
    
    elif command == "type-check":
        success = run_command("mypy ladder/", "Running mypy")
        if success:
            print("✅ Type checking passed")
    
    elif command == "clean":
        print("Cleaning build artifacts...")
        import shutil
        
        paths_to_clean = [
            "build", "dist", "*.egg-info", "__pycache__",
            ".pytest_cache", ".mypy_cache", ".coverage", "htmlcov"
        ]
        
        for pattern in paths_to_clean:
            for path in Path(".").rglob(pattern):
                if path.exists():
                    if path.is_dir():
                        shutil.rmtree(path)
                        print(f"  Removed directory: {path}")
                    else:
                        path.unlink()
                        print(f"  Removed file: {path}")
        
        print("✅ Cleanup complete")
    
    elif command == "all":
        print("Running full quality check pipeline...")
        
        steps = [
            ("format", "Formatting code"),
            ("lint", "Linting"),
            ("type-check", "Type checking"),
            ("test", "Testing")
        ]
        
        for step_cmd, description in steps:
            print(f"\n📋 {description}...")
            if step_cmd == "format":
                success1 = run_command("black ladder/ tests/", "Running black")
                success2 = run_command("isort ladder/ tests/", "Running isort") 
                step_success = success1 and success2
            elif step_cmd == "lint":
                step_success = run_command("flake8 ladder/ tests/", "Running flake8")
            elif step_cmd == "type-check":
                step_success = run_command("mypy ladder/", "Running mypy")
            elif step_cmd == "test":
                step_success = run_command("pytest", "Running tests")
            
            if not step_success:
                print(f"❌ {description} failed")
                success = False
                break
            else:
                print(f"✅ {description} passed")
        
        if success:
            print("\n🎉 All checks passed! Your code is ready.")
    
    else:
        print(f"❌ Unknown command: {command}")
        success = False
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()