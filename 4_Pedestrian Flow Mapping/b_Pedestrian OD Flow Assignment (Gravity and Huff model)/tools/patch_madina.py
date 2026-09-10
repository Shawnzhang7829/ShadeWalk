import re, pathlib
p = pathlib.Path(r"D:/Claude/UNA/madina-main/src/madina/zonal/network_utils.py")
src = p.read_text(encoding="utf-8")
new = re.sub(r"fastpath=True,\s*", "", src)
p.write_text(new, encoding="utf-8")
print("patched. remaining fastpath:", new.count("fastpath"))
