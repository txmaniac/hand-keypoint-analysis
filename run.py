import os
import subprocess
import sys
from pathlib import Path

def print_status(message):
    print(f"\n[+] {message}")

def print_error(message):
    print(f"\n[-] ERROR: {message}")

def main():
    print("==============================================")
    print("  Holistic Biomechanical Analytics Engine")
    print("  [Autonomous Astral Sandbox Bootstrapper]")
    print("==============================================\n")

    root_dir = Path(__file__).parent.resolve()
    bootstrap_dir = root_dir / ".bootstrap_uv"
    venv_dir = root_dir / ".venv"
    
    # OS execution bindings
    if os.name == 'nt':
        uv_executable = bootstrap_dir / "Scripts" / "uv.exe"
        venv_python = venv_dir / "Scripts" / "python.exe"
    else:
        uv_executable = bootstrap_dir / "bin" / "uv"
        venv_python = venv_dir / "bin" / "python"

    # Step 1: Bootstrap `uv` in a completely hidden throwaway environment natively
    if not uv_executable.exists():
        print_status("Bootstrapping `uv` (Ultra-fast Python Manager)...")
        try:
            import venv
            builder = venv.EnvBuilder(with_pip=True)
            builder.create(bootstrap_dir)
            target_py = bootstrap_dir / "Scripts" / "python.exe" if os.name == 'nt' else bootstrap_dir / "bin" / "python"
            subprocess.check_call([str(target_py), "-m", "pip", "install", "uv"], stdout=subprocess.DEVNULL)
        except Exception as e:
            print_error(f"Failed to bootstrap UV engine: {e}")
            sys.exit(1)

    # Step 2: Use `uv` to autonomously download Python 3.11 and lock the Sandbox permanently
    if not venv_dir.exists():
        print_status("Commanding Astral UV to provision isolated Python 3.11 Runtime (this downloads python dynamically)...")
        try:
            subprocess.check_call([str(uv_executable), "venv", str(venv_dir), "--python", "3.11"])
        except subprocess.CalledProcessError as e:
            print_error(f"Astral UV failed to provision Python 3.11: {e}")
            sys.exit(1)
    else:
        print_status("Isolated Python 3.11 Sandbox verified.")

    # Step 3: Install Requirements strictly via `uv pip` for blazing fast parallel resolution
    req_file = root_dir / "requirements.txt"
    print_status("Locking dependencies via ultra-fast parallel installation...")
    try:
        env = os.environ.copy()
        env["VIRTUAL_ENV"] = str(venv_dir)  # Tell UV which venv to target
        subprocess.check_call([str(uv_executable), "pip", "install", "-r", str(req_file)], env=env)
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to cleanly install python dependencies: {e}")
        sys.exit(1)

    # Step 4: Boot Engine Interface securely from inside the isolated sandbox
    print_status("Booting up the Native Streamlit Sandbox...")
    app_script = root_dir / "app_with_yolo.py"
    try:
        subprocess.run([str(venv_python), "-m", "streamlit", "run", str(app_script)])
    except KeyboardInterrupt:
        print_status("Received manual override. Shutting down the Sandbox. Goodbye!")

if __name__ == "__main__":
    main()
