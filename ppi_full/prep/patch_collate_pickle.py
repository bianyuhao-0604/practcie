p = "data.py"
b = open(p, "rb").read()
a1 = b"class FeatureStore:"
a2 = b"def make_collate(store):"
a3 = b"    return collate"
if b.count(b"functools.partial(_collate_dispatch, store)") == 1:
    print("ALREADY PATCHED - skip")
elif b.count(a1) == 1 and b.count(a2) == 1 and b.count(a3) == 1:
    nb = b.replace(a1, b"import functools\n\n\n" + a1)
    nb = nb.replace(a2, b"def _collate_dispatch(store, items):\n    return make_collate(store)(items)\n\n\n" + a2)
    nb = nb.replace(a3, b"    return functools.partial(_collate_dispatch, store)")
    open(p, "wb").write(nb)
    print("PATCHED: collate closure -> functools.partial(module-level dispatch)")
else:
    raise SystemExit(f"AMBIGUOUS: a1 x{b.count(a1)}, a2 x{b.count(a2)}, a3 x{b.count(a3)} - stop")
compile(open(p, "rb").read(), p, "exec")
print("COMPILE OK")
