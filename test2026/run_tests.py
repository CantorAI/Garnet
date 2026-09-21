# SPDX-FileCopyrightText: 2024-2026 CantorAI Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import glob
import json
import time
import subprocess
try:
    import yaml
except ImportError:
    print("PyYAML is required. Please install it with 'pip install pyyaml'.")
    sys.exit(1)

def discover_xlang():
    # Try common paths for xlang executable based on the workspace structure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    paths = [
        os.path.join(workspace_root, "xlang", "out", "build", "x64-Debug", "bin", "xlang.exe"),
        os.path.join(workspace_root, "xlang", "out", "build", "x64-Release", "bin", "xlang.exe"),
        os.path.join(workspace_root, "xlang", "build_linux", "bin", "xlang")
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    return "xlang" # Fallback to PATH

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.yaml")
    
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        sys.exit(1)
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    # Set environment variables
    env = os.environ.copy()
    if "env_vars" in config:
        for k, v in config["env_vars"].items():
            env[k] = str(v)
            print(f"Set ENV: {k} = {v}")
            
    timeout = config.get("timeout", 30)
    
    # Test Discovery
    includes = config.get("test_discovery", {}).get("include", [])
    excludes = config.get("test_discovery", {}).get("exclude", [])
    
    test_dirs = set()
    for inc in includes:
        pattern = os.path.join(script_dir, inc)
        for d in glob.glob(pattern):
            if os.path.isdir(d):
                test_dirs.add(os.path.abspath(d))
                
    for exc in excludes:
        pattern = os.path.join(script_dir, exc)
        for d in glob.glob(pattern):
            d_abs = os.path.abspath(d)
            if d_abs in test_dirs:
                test_dirs.remove(d_abs)
                
    test_dirs = sorted(list(test_dirs))
    
    xlang_exe = discover_xlang()
    
    results = []
    passed_count = 0
    failed_count = 0
    
    print(f"\nDiscovered {len(test_dirs)} test directories.")
    
    for d in test_dirs:
        dir_name = os.path.basename(d)
        
        # Discover all test files in the directory
        test_files = []
        for ext in ["*.py", "*.x"]:
            # Match both "test.py" and "test_something.py"
            test_files.extend(glob.glob(os.path.join(d, f"test{ext}")))
            test_files.extend(glob.glob(os.path.join(d, f"test_{ext}")))
        
        test_files = sorted(list(set(test_files)))
        
        if not test_files:
            print(f"Directory {dir_name}: SKIPPED (No test files found)")
            continue
            
        for tf in test_files:
            file_name = os.path.basename(tf)
            test_name = f"{dir_name}/{file_name}"
            print(f"Running: {test_name} ...", end=" ", flush=True)
            
            cmd = [sys.executable, tf] if tf.endswith(".py") else [xlang_exe, tf]
            
            start_time = time.time()
            try:
                res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout, cwd=d)
                exec_time = time.time() - start_time
                
                output = res.stdout + "\n" + res.stderr
                status = "PASS" if res.returncode == 0 else "FAIL"
                
                if status == "PASS":
                    passed_count += 1
                    print("PASS")
                else:
                    failed_count += 1
                    print(f"FAIL (Exit code {res.returncode})")
                    
                results.append({
                    "name": test_name,
                    "status": status,
                    "exit_code": res.returncode,
                    "execution_time_sec": round(exec_time, 2),
                    "output": output.strip()
                })
                
            except subprocess.TimeoutExpired as e:
                exec_time = time.time() - start_time
                failed_count += 1
                print("TIMEOUT")
                output = ""
                if e.stdout: output += e.stdout.decode('utf-8', errors='ignore') if isinstance(e.stdout, bytes) else e.stdout
                if e.stderr: output += "\n" + (e.stderr.decode('utf-8', errors='ignore') if isinstance(e.stderr, bytes) else e.stderr)
                
                results.append({
                    "name": test_name,
                    "status": "TIMEOUT",
                    "exit_code": -1,
                    "execution_time_sec": round(exec_time, 2),
                    "output": output.strip()
                })
            except Exception as e:
                exec_time = time.time() - start_time
                failed_count += 1
                print(f"ERROR ({str(e)})")
                results.append({
                    "name": test_name,
                    "status": "ERROR",
                    "exit_code": -1,
                    "execution_time_sec": round(exec_time, 2),
                    "output": str(e)
                })
            
    # Write JSON report
    report = {
        "summary": {
            "total": len(results),
            "passed": passed_count,
            "failed": failed_count
        },
        "tests": results
    }
    
    report_path = os.path.join(script_dir, "results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
        
    print(f"\nTest run complete. Summary: {passed_count} Passed, {failed_count} Failed.")
    print(f"Detailed results written to {report_path}")
    
    if failed_count > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
