p = "data.py"
b = open(p, "rb").read()
old1 = b'                src = getattr(store, k)'
new1 = (b'                src = getattr(store, k)\n'
        b'                if src is None:\n'
        b'                    if k == "seq":\n'
        b'                        raise FileNotFoundError("seq.npy missing in " + store.dir + " - run build_seq_store.py")\n'
        b'                    d[k] = np.zeros((B, Lmax, {"str": 1024, "surf": 80}[k]), np.float32)\n'
        b'                    continue')
old2 = b'                    iface[jj, :Lc] = np.asarray(store.iface[ix, :Lc], np.float32)'
new2 = (b'                    if store.iface is not None:\n'
        b'                        iface[jj, :Lc] = np.asarray(store.iface[ix, :Lc], np.float32)')

if b.count(old1) == 0 and b.count(b'if src is None:') == 1:
    print("ALREADY PATCHED - skip")
else:
    n1, n2 = b.count(old1), b.count(old2)
    if n1 != 1 or n2 != 1:
        raise SystemExit(f"AMBIGUOUS: patch1 x{n1}, patch2 x{n2} - stop")
    open(p, "wb").write(b.replace(old1, new1).replace(old2, new2))
    print("PATCHED: side() zero-fill for missing str/surf + iface guard")
compile(open(p, "rb").read(), p, "exec")
print("COMPILE OK")
