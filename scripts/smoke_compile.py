import compileall, sys

ok = compileall.compile_dir(".", force=True, quiet=1)
if not ok:
    sys.exit("Compile errors detected. Fix before running.")
print("OK: All .py files compiled successfully.")
