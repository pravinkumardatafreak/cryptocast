import sys
import os
import glob
import traceback

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from streamlit.testing.v1 import AppTest

# Test app.py first
print("==================================================")
print("TESTING app.py")
print("==================================================")
try:
    at = AppTest.from_file(os.path.join(PROJECT_DIR, "app.py"), default_timeout=30)
    at.run()
    if at.exception:
        print("EXCEPTION in app.py:")
        for exc in at.exception:
            print(exc)
    else:
        print("SUCCESS: app.py loaded without uncaught exceptions.")
except Exception as e:
    print("FAILED running app.py:", e)
    traceback.print_exc()

pages_dir = os.path.join(PROJECT_DIR, 'pages')
page_files = sorted(glob.glob(os.path.join(pages_dir, '*.py')))

for page_file in page_files:
    basename = os.path.basename(page_file)
    print("\n==================================================")
    print(f"TESTING PAGE: {basename}")
    print("==================================================")
    try:
        at = AppTest.from_file(page_file, default_timeout=30)
        at.run()
        if at.exception:
            print(f"EXCEPTION in {basename}:")
            for exc in at.exception:
                print(exc.value)
                print(exc.stack_trace)
        else:
            print(f"SUCCESS: {basename} rendered without uncaught exceptions.")
    except Exception as e:
        print(f"FAILED running {basename}:", e)
        traceback.print_exc()

print("\nFinished all page runtime tests.")
