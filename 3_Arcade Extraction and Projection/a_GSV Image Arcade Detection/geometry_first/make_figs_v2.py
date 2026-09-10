"""Regenerate the standard step2_projection figure set from the v2 vectors (precise/vectors),
writing PNGs to detection_geomfirst/figs_v2/. Original scripts are copied & path-patched only —
no logic changes. Run in pyenv."""
import subprocess, sys
from pathlib import Path
ROOT=Path(r"D:\Claude\SVI_FFW"); SRC=ROOT/"output"/"step2_projection"
GF=ROOT/"output"/"detection_geomfirst"; FIG=GF/"figs_v2"; SCR=FIG/"_scripts"
FIG.mkdir(exist_ok=True); SCR.mkdir(exist_ok=True)
VEC='ROOT/"output"/"detection_geomfirst"'   # v2 vectors now live at the detection_geomfirst root (the user flattened the directory)
FIGL='ROOT/"output"/"detection_geomfirst"/"figs_v2"'
MAPVIZ=r'_s.path.insert(0, r"D:\Claude\SVI_FFW\output\step2_projection")'

PATCH={
 "viz_step2.py":[
   ('OUT=ROOT/"output"/"step2_projection"', f'OUT={VEC}\nFIG={FIGL}'),
   ('plt.savefig(OUT/f"viz_step2_{city}.png"', 'plt.savefig(FIG/f"viz_step2_{city}.png"')],
 "viz_segments_vs_runs.py":[
   ('P=ROOT/"output"/"step2_projection"', f'P={VEC}\nFIG={FIGL}'),
   ('_s.path.insert(0, str(P))', MAPVIZ),
   ('plt.savefig(P/name', 'plt.savefig(FIG/name')],
 "viz_closeup.py":[
   ('P=ROOT/"output"/"step2_projection"', f'P={VEC}\nFIG={FIGL}'),
   ('OUT=P/"viz_closeup_fixed.png"', 'OUT=FIG/"viz_closeup_fixed.png"')],
 "viz_verify_rules.py":[
   ('P=ROOT/"output"/"step2_projection"', f'P={VEC}\nFIG={FIGL}'),
   ('OUT=P/"viz_verify_rules.png"', 'OUT=FIG/"viz_verify_rules.png"')],
 "viz_pipeline_flow.py":[
   ('P=ROOT/"output"/"step2_projection"', f'P={VEC}\nFIG={FIGL}'),
   ('OUT=P/"viz_pipeline_flow.png"', 'OUT=FIG/"viz_pipeline_flow.png"')],
}
# fullcity_map.py lives in output/, slightly different constants
FULL=[
 ('OUT=ROOT/"output"/"step2_projection"', f'OUT={VEC}\nFIG={FIGL}'),
 ('plt.suptitle("City-wide Arcade Distribution"', 'plt.suptitle("City-wide Arcade Distribution (v2 geometry-first)"'),
 ('for p in [ROOT/"output"/"fullcity_arcade_map.png", ROOT/"output"/"step2_projection"/"fullcity_arcade_map.png"]:',
  'for p in [FIG/"fullcity_arcade_map.png"]:'),
]

def patch(src,reps,dst):
    t=src.read_text(encoding="utf-8")
    for a,b in reps:
        assert a in t, f"pattern not found in {src.name}: {a[:50]}"
        t=t.replace(a,b)
    dst.write_text(t,encoding="utf-8")

jobs=[]
for name,reps in PATCH.items():
    patch(SRC/name,reps,SCR/name)
patch(ROOT/"output"/"fullcity_map.py",FULL,SCR/"fullcity_map.py")

RUNS=[("viz_step2.py",["sg"]),("viz_step2.py",["bo"]),
      ("viz_segments_vs_runs.py",["sg"]),("viz_segments_vs_runs.py",["bo"]),
      ("viz_closeup.py",[]),("viz_verify_rules.py",[]),("viz_pipeline_flow.py",[]),
      ("fullcity_map.py",[])]
for name,args in RUNS:
    print(f"=== {name} {' '.join(args)} ===",flush=True)
    r=subprocess.run([sys.executable,str(SCR/name),*args],capture_output=True,text=True,encoding="utf-8",errors="replace")
    print((r.stdout or "").strip()[-300:])
    if r.returncode!=0: print("STDERR:",(r.stderr or "").strip()[-500:])
print("\n[all figs_v2 done]")
