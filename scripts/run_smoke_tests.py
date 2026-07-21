from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
test_path = ROOT / "tests" / "test_integrated_model.py"
spec = importlib.util.spec_from_file_location("integrated_tests", test_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {test_path}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
tests = [getattr(module, name) for name in sorted(dir(module)) if name.startswith("test_")]
for test in tests:
    test()
    print(f"PASS {test.__name__}")
print(f"{len(tests)} smoke tests passed")
