res = []
# --- model.py: Reim2D stack -> cat ---
p = "model.py"; b = open(p, "rb").read()
old = b"return torch.stack(logits)[:, None], None"
new = b"return torch.cat(logits, dim=0)[:, None], None"
if b.count(old) == 1:
    open(p, "wb").write(b.replace(old, new)); res.append("model.py: stack->cat PATCHED")
elif b.count(new) == 1:
    res.append("model.py: already patched")
else:
    raise SystemExit(f"model.py AMBIGUOUS x{b.count(old)}")

# --- engine.py: 两个 BCE 调用点 squeeze(-1) ---
p = "engine.py"; b = open(p, "rb").read()
e1o = b'loss = F.binary_cross_entropy_with_logits(out1["logits"].float(), t1)'
e1n = b'loss = F.binary_cross_entropy_with_logits(out1["logits"].float().squeeze(-1), t1)'
e2o = b'F.binary_cross_entropy_with_logits(out2["logits"].float(), t2)'
e2n = b'F.binary_cross_entropy_with_logits(out2["logits"].float().squeeze(-1), t2)'
if b.count(e1o) == 1 and b.count(e2o) == 1:
    open(p, "wb").write(b.replace(e1o, e1n).replace(e2o, e2n)); res.append("engine.py: 2x squeeze PATCHED")
elif b.count(e1n) == 1 and b.count(e2n) == 1:
    res.append("engine.py: already patched")
else:
    raise SystemExit(f"engine.py AMBIGUOUS e1 x{b.count(e1o)} e2 x{b.count(e2o)}")

compile(open("model.py", "rb").read(), "model.py", "exec")
compile(open("engine.py", "rb").read(), "engine.py", "exec")
print(" | ".join(res), "| COMPILE OK x2")
