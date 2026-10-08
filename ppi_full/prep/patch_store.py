p = "features/store.py"
b = open(p, "rb").read()
old, new = b"shape=(self.n, dim))", b"shape=(self.n, *dim))"
n = b.count(old)
if n == 0:
    assert b.count(new) == 1, "patch pattern missing — stop"
    print("ALREADY PATCHED — skip")
elif n == 1:
    open(p, "wb").write(b.replace(old, new))
    print("PATCHED: PROT_LAYOUT memmap shape (self.n, dim) -> (self.n, *dim)")
else:
    raise SystemExit(f"AMBIGUOUS: {n} matches — stop")
b2 = open(p, "rb").read()
assert b2.count(new) == 1 and b2.count(old) == 0
print("VERIFY OK")
