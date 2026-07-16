import sys
from pathlib import Path

# Insert src directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# Import test functions
from tests.test_data_validation import test_check_missing_values, test_check_duplicates, test_check_value_ranges
from tests.test_preprocessing import test_build_preprocessor
from tests.test_models import test_model_fitting
from tests.test_pipeline import test_pipeline_config

print("=== RUNNING AUTOMATED UNIT TESTS ===")
tests = [
    ("test_check_missing_values", test_check_missing_values),
    ("test_check_duplicates", test_check_duplicates),
    ("test_check_value_ranges", test_check_value_ranges),
    ("test_build_preprocessor", test_build_preprocessor),
    ("test_model_fitting", test_model_fitting),
    ("test_pipeline_config", test_pipeline_config)
]

failures = 0
for name, func in tests:
    try:
        func()
        print(f"  [PASS] {name}")
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        import traceback
        traceback.print_exc()
        failures += 1

print(f"\nTest run complete. Failures: {failures} / {len(tests)}")
sys.exit(1 if failures > 0 else 0)
