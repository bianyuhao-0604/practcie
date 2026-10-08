p = "data.py"
b = open(p, "rb").read()
a_d = b"def _collate_dispatch(store, items):\n    return make_collate(store)(items)"
n_d = (b"_WORKER_STORE = None\n\n\ndef _collate_dispatch(feat_dir, items):\n"
       b"    global _WORKER_STORE\n"
       b"    if _WORKER_STORE is None or _WORKER_STORE.dir != feat_dir:\n"
       b"        _WORKER_STORE = FeatureStore(feat_dir)\n"
       b"    return make_collate(_WORKER_STORE, _raw=True)(items)")
a_m = b"def make_collate(store):"
n_m = b"def make_collate(store, _raw=False):"
a_r = b"    return functools.partial(_collate_dispatch, store)"
n_r = (b"    if _raw:\n        return collate\n"
       b"    return functools.partial(_collate_dispatch, store.dir)")

if b.count(b"_WORKER_STORE") >= 1 and b.count(a_d) == 0:
    print("ALREADY PATCHED v2 - skip")
elif b.count(a_d) == 1 and b.count(a_m) == 1 and b.count(a_r) == 1:
    nb = b.replace(a_d, n_d).replace(a_m, n_m).replace(a_r, n_r)
    open(p, "wb").write(nb)
    print("PATCHED v2: payload=feat_dir string (109B), recursion fixed via _raw channel")
else:
    raise SystemExit(f"AMBIGUOUS: a_d x{b.count(a_d)}, a_m x{b.count(a_m)}, a_r x{b.count(a_r)} - stop")
compile(open(p, "rb").read(), p, "exec")
print("COMPILE OK")
