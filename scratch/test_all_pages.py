import sys
import os
import glob
import traceback

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

pages_dir = os.path.join(PROJECT_DIR, 'pages')
page_files = sorted(glob.glob(os.path.join(pages_dir, '*.py')))

print(f"Found {len(page_files)} pages to check.")

for page_file in page_files:
    basename = os.path.basename(page_file)
    print(f"\n--- Syntax & Compile Check: {basename} ---")
    try:
        with open(page_file, 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, page_file, 'exec')
        print(f"PASSED compilation: {basename}")
    except Exception as e:
        print(f"FAILED compilation in {basename}:")
        traceback.print_exc()

print("\nFinished compile checks.")
