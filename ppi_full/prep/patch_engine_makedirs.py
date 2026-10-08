p = "engine.py"
b = open(p, "rb").read()
old = b'    save_json({"model":'
new = b'    os.makedirs(os.path.join(paths.out_dir, tag, f"seed{seed}"), exist_ok=True)\n' + old
if b.count(new) == 1 and b.count(old) == 0:
    print("ALREADY PATCHED - skip")
elif b.count(old) == 1:
    open(p, "wb").write(b.replace(old, new))
    print("PATCHED: makedirs before cfg.json save")
else:
    raise SystemExit(f"AMBIGUOUS: anchor x{b.count(old)} - stop")
