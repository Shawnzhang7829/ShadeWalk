# -*- coding: utf-8 -*-
"""Generate the "paper-routing" build nav_app_paper.html (+_en; the existing versions are not overwritten).
Background: the core Coolest navigation (coolDetour: len*((1-shade)+lambda), most-shaded route within detour <= tau) and the
pedestrian-flow distribution (step4_4e: lambda sweep, most-shaded route within detour <= tau) already use only "shade fraction +
detour ratio cap", consistent with the paper; the only place with unsupported weights is the topology-graph visualisation
W1 (0.2 crowding + 0.3 (1-shade) + 0.3 detour + 0.2 continuous sun) / W2 (0.4 crowding + 0.6 (1-shade)).
This script replaces the topology-graph resistance with the paper's Fig4 definition: street resistance rho=(1-shade)+lambda (lambda=0.2),
path weight omega=sum(len*rho), coolest=min omega; crowding / shade continuity removed. Post-processes the existing nav_app.html (ZH);
the English version is produced by _make_en. pyenv."""
WEB=r"D:\Claude\SVI_FFW\output\step5_nav_webapp\webapp"
h=open(f"{WEB}\\nav_app.html",encoding="utf-8").read()

TOPOW_OLD="const TOPOW={crowd:0.20,shade:0.30,detour:0.30,sunrun:0.20,s_crowd:0.40,s_shade:0.60};  // expert-set weights (no canonical 4-factor weights in literature; shade-related dominates at 50%/60%, cf. CoolWalks single-alpha paradigm)"
TOPOW_NEW="const TOPOW={};  // paper routing (Fig4): street resistance rho=(1-shade)+lambda, path weight omega=sum len*rho; coolest=min omega. shade coverage + detour cap only."

TOPOMET_OLD="""function topoMet(g){ // per-path metrics + composite resistance W1 (crowding uses the current Detour-limit flow field)
 let F=flowCur(),ms=g.cands.map(function(r){let cr=0,shm=0,run=0,mx=0;
  for(let k=0;k<r.eis.length;k++){let ei=r.eis[k];cr+=F[ei];let sv=esf(ei);shm+=EL[ei]*sv/100;if(sv<50){run+=EL[ei];if(run>mx)mx=run;}else run=0;}
  return {r:r,len:r.len,shade:shm/(r.len||1),crowd:cr/(r.eis.length||1),det:r.len/(g.s.len||1),sunrun:mx/(r.len||1)};});
 let cmax=1;ms.forEach(function(m){if(m.crowd>cmax)cmax=m.crowd;});
 ms.forEach(function(m){m.crowdN=m.crowd/cmax;m.detN=Math.max(0,Math.min(1,(m.det-1)/Math.max(DETOUR-1,0.05)));
  m.W1=TOPOW.crowd*m.crowdN+TOPOW.shade*(1-m.shade)+TOPOW.detour*m.detN+TOPOW.sunrun*m.sunrun;});
 ms.sort(function(a,b){return a.W1-b.W1;});return ms;}"""
TOPOMET_NEW="""function topoMet(g){ // paper routing (Fig4): path weight omega = sum len*((1-shade)+lambda); coolest = min omega. shade+detour only (no crowding/continuity).
 let ms=g.cands.map(function(r){let shm=0,om=0;
  for(let k=0;k<r.eis.length;k++){let ei=r.eis[k],sv=esf(ei);shm+=EL[ei]*sv/100;om+=EL[ei]*((1-sv/100)+PLAM);}
  return {r:r,len:r.len,shade:shm/(r.len||1),det:r.len/(g.s.len||1),W1:om};});
 ms.sort(function(a,b){return a.W1-b.W1;});return ms;}"""

TOPOST_OLD="""  let len=0,shm=0,cr=0,F=flowCur();chain.forEach(function(e2){len+=EL[e2];shm+=EL[e2]*esf(e2)/100;cr+=F[e2];});
  st2.push({nds:nds,len:len,shade:shm/(len||1),crowd:cr/(chain.length||1)});});});
 let cmax=1;st2.forEach(function(s2){if(s2.crowd>cmax)cmax=s2.crowd;});
 st2.forEach(function(s2){s2.W2=TOPOW.s_crowd*(s2.crowd/cmax)+TOPOW.s_shade*(1-s2.shade);});"""
TOPOST_NEW="""  let len=0,shm=0;chain.forEach(function(e2){len+=EL[e2];shm+=EL[e2]*esf(e2)/100;});
  st2.push({nds:nds,len:len,shade:shm/(len||1)});});});
 st2.forEach(function(s2){s2.W2=(1-s2.shade)+PLAM;});   // paper Fig4: rho = (1-shade)+lambda"""

# g1svg tooltip: drop crowding / continuous sun, show omega
G1TIP_OLD="% · 拥挤 '+Math.round(m.crowd)+' · 绕行 '+m.det.toFixed(2)+'x · 最长连续日晒 '+Math.round(m.sunrun*m.len)+'m · W='+m.W1.toFixed(3)+'"
G1TIP_NEW="% · 绕行 '+m.det.toFixed(2)+'x · ω='+Math.round(m.W1)+'"
# g1svg P label: W1.toFixed(2) -> integer omega
G1LBL_OLD="' '+m.W1.toFixed(2)+'</text>"
G1LBL_NEW="' ω'+Math.round(m.W1)+'</text>"
# g2svg tooltip: drop crowding, show rho
G2TIP_OLD="· 拥挤 '+Math.round(s2.crowd)+' · W='+s2.W2.toFixed(3)+'"
G2TIP_NEW="· ρ='+s2.W2.toFixed(2)+'"
# comparison card: best-path resistance W -> best-path weight omega (integer)
CARD_OLD="最优路阻力 W '+b.W.toFixed(3)+' → <b>'+c.W.toFixed(3)+'</b> ('+(c.W>b.W?'+':'')+(c.W-b.W).toFixed(3)+')"
CARD_NEW="最优路权重 ω '+Math.round(b.W)+' → <b>'+Math.round(c.W)+'</b> ('+(c.W>b.W?'+':'')+Math.round(c.W-b.W)+')"
# Graph 1 title
T1_OLD="图1 · 路径级:点=OD,边=可行路径(互不交叉;越宽阻力越小;W=0.2拥挤+0.3(1−遮荫)+0.3绕行+0.2连续日晒)"
T1_NEW="图1 · 路径级:点=OD,边=可行路径(互不交叉;越宽=权重越小=越优;路径权重 ω=Σℓ×((1−遮荫)+λ),λ='+PLAM+',coolest=min ω)"
# Graph 2 title
T2_OLD="图2 · 街道级:点=OD+道路交叉口,边=街道(越宽阻力越小;W=0.4拥挤+0.6(1−遮荫))"
T2_NEW="图2 · 街道级:点=OD+道路交叉口,边=街道(越宽阻力越小;街道路权 ρ=(1−遮荫)+λ,λ='+PLAM+')"
# per-path metric rows: drop crowding / continuous sun, show omega
MET_OLD="'m · 遮荫'+Math.round(m.shade*100)+'% · 拥挤'+Math.round(m.crowd)+' · 绕行'+m.det.toFixed(2)+'× · 连续日晒'+Math.round(m.sunrun*100)+'% · <b>W='+m.W1.toFixed(3)+'</b>'"
MET_NEW="'m · 遮荫'+Math.round(m.shade*100)+'% · 绕行'+m.det.toFixed(2)+'× · <b>ω='+Math.round(m.W1)+'</b>'"
# footnote: switch to the paper-routing explanation
FOOT_OLD="候选=λ扫描+迭代惩罚多样化(≤8 条代表性可行路径,穷举为指数级不可行);拥挤度=沿路人流均值(按候选集内最大值归一);连续日晒=最长连续日晒段长度占比(边遮荫<50% 记日晒);点击弧线或行可在地图高亮(紫虚线)。权重为专家设定值(文献无公认四因子权重;遮荫相关合计 50%/60%,参照 CoolWalks 单参数避晒范式)。"
FOOT_NEW="候选=λ扫描+迭代惩罚多样化(≤8 条代表性可行路径);阻力=论文口径(Fig4):街道路权 ρ=(1−遮荫)+λ(λ='+PLAM+',滑杆可调),路径权重 ω=Σℓ×ρ,coolest=min ω;仅阴影覆盖率+绕行率上限,不含拥挤度/遮荫连续性;点击弧线或行可在地图高亮(紫虚线)。"
# topology window header + page title flagged as paper routing
HDR_OLD="🕸 拓扑图 · OD 可行路径"
HDR_NEW="🕸 拓扑图 · OD 可行路径(ρ/ω)"
TITLE_OLD="<title>ShadeWalk</title>"
TITLE_NEW="<title>ShadeWalk · paper routing</title>"

# ===== paper-style Coolest: single Dijkstra with fixed lambda (min omega); the slider now selects the lambda level and shows the sun/shade exchange rate lambda/(1+lambda) =====
# panel heading: detour limit -> sun/shade exchange rate
SEC_OLD='<div class="sec" style="margin:9px 0 4px">绕行阈值 Detour limit</div>'
SEC_NEW=('<div class="sec" style="margin:9px 0 4px">遮荫回报率 Shade reward(η)</div>\n'
 '  <div style="font-size:10.5px;color:#6b6a63;margin-bottom:5px;line-height:1.55">当 λ 取默认值 0.2 时:遮荫回报率 η=λ/(1+λ)=<b>1/6</b>,即绕行只有在至少这一比例的额外路程转化为减少的日晒时才划算;<b>日晒 1 米 = 荫下 6.0 米</b>。例:1000 m 的路多绕 100 m,须至少少晒 17 m 才值得。</div>')
# slider: lambda levels (formerly the 4 detour levels); the right side shows lambda and the exchange rate
SLD_OLD='<input type="range" id="detTh" min="0" max="3" step="1" value="2" style="flex:1"><span id="detThV" style="min-width:40px;text-align:right;color:#0F6E56;font-weight:600">1.5×</span>'
SLD_NEW='<input type="range" id="detTh" min="0" max="16" step="1" value="9" style="flex:1"><span id="detThV" style="min-width:104px;text-align:right;color:#0F6E56;font-weight:600">λ=0.2 · η=16.7%</span>'
# DETOUR declaration -> also declare PLAM (paper lambda)
DEC_OLD="let DETOUR=1.5;   // detour limit (slider): cool path length <= shorte"
DEC_NEW="let PLAM=0.2;     // paper lambda (slider): coolest = min omega = sum len*((1-shade)+PLAM)\nlet DETOUR=1.5;   // kept only for topology candidate filtering; NOT used by the map Coolest in this paper build. detour limit: cool path length <= shorte"
# route: coolDetour -> single dij with fixed lambda
RT_OLD="let s=dij(O,Dst,'short'),c=coolDetour(O,Dst,s);tip(0);"
RT_NEW="let s=dij(O,Dst,'short'),c=dij(O,Dst,'cool',PLAM);tip(0);   // paper: fixed-lambda min omega (no detour cap)"
# slider handler: change lambda, update the exchange-rate text + approximate the flow field by the nearest tau
DTH_OLD="var DTH=[1,1.2,1.5,2];document.getElementById('detTh').oninput=function(){DETOUR=DTH[+this.value];document.getElementById('detThV').textContent=DETOUR+'×';if(colorMode==='flow')redrawNet();if(O>=0&&Dst>=0)route();};"
DTH_NEW=("var PLS=[0.005,0.01,0.02,0.04,0.06,0.09,0.12,0.15,0.18,0.2,0.25,0.35,0.5,0.7,1.0,1.5,3.0];  // SI Table lambda sweep (17 finite levels; lambda=inf is the Shortest route)\n"
 "function lamUpd(){var rt=PLAM/(1+PLAM);\n"
 " document.getElementById('detThV').textContent='λ='+PLAM+' · η='+(rt*100).toFixed(1)+'%';\n"
 " var eq=document.getElementById('lamEq'),rr=document.getElementById('lamRt');\n"
 " if(eq)eq.textContent=(1+1/PLAM).toFixed(1);if(rr)rr.textContent=(rt*100).toFixed(1)+'%';\n"
 " var ex=document.getElementById('lamEx');if(ex)ex.textContent=Math.round(100*rt);}\n"
 "document.getElementById('detTh').oninput=function(){PLAM=PLS[+this.value];lamUpd();\n"
 " if(colorMode==='flow'||colorMode==='rho')redrawNet();if(O>=0&&Dst>=0)route();if(typeof topoOpen==='function'&&topoOpen())topoBuild(true);};lamUpd();")
# flow legend: label lambda and the approximation
LEG_OLD="'步行人流量(避热 τ='+DETOUR+'×)'"
LEG_NEW="'步行人流量(论文口径 λ='+PLAM+')'"
# flowCur: use the fixed-lambda flow fields (step4_4f); fall back to the nearest level when one is missing
FC_OLD="function flowCur(){return FT[DETOUR]||EF;}  // pedestrian-flow field matching the current Detour limit; falls back to flow_short"
FC_NEW=("function flowCur(){if(!D.flam)return EF;var k=String(PLAM).replace('.','p');if(D.flam[k])return D.flam[k];\n"
 " var ks=Object.keys(D.flam),best=null,bd=1e9;ks.forEach(function(s){var v=parseFloat(s.replace('p','.'));var dd=Math.abs(Math.log(v)-Math.log(PLAM));if(dd<bd){bd=dd;best=s;}});\n"
 " return best?D.flam[best]:EF;}  // paper build: fixed-lambda flow fields (step4_4f); nearest lambda level if exact one is absent")
# ===== new rho/omega distribution mode in Network view: colour the whole network by edge resistance rho=(1-sigma)+lambda, redrawn live with the lambda slider =====
BTN_OLD='<button class="btn" id="cShade">遮荫率</button>'
BTN_NEW='<button class="btn" id="cShade">遮荫率</button><button class="btn" id="cRho">路段成本 ωᵢ</button>'
HND_OLD="document.getElementById('cPlain').onclick=function(){colorMode='plain';pick(this,['cShade','cFlow','cPlain','cNone']);redrawNet();};"
HND_NEW=("document.getElementById('cPlain').onclick=function(){colorMode='plain';pick(this,['cShade','cFlow','cRho','cPlain','cNone']);redrawNet();};\n"
 "document.getElementById('cRho').onclick=function(){colorMode='rho';pick(this,['cShade','cFlow','cRho','cPlain','cNone']);ensureNetVisible&&0;redrawNet();cleanView();tip('路段成本 ωᵢ=ℓ×ρ 分布(ρ=(1−σ)+λ,λ='+PLAM+'):深=成本高(长且晒),浅=成本低(短或荫);最遮荫路 = min ω=Σωᵢ');};")
PICK_OLD="pick(this,['cShade','cFlow','cPlain','cNone']);"
PICK_NEW="pick(this,['cShade','cFlow','cRho','cPlain','cNone']);"
BIN_OLD="function binFC(mode,srcset){let nb=mode==='shade'?8:6,"
BIN_NEW="function binFC(mode,srcset){let nb=(mode==='shade'||mode==='rho')?8:6,"
BINB_OLD="let b=mode==='shade'?Math.min(7,Math.floor(esf(i)/100*8)):Math.min(5,Math.floor(Math.log(EFx[i]+1)/lf*6));"
BINB_NEW=("let b;if(mode==='rho'){var wi=EL[i]*((1-esf(i)/100)+PLAM);b=Math.max(0,Math.min(7,Math.floor(Math.log(wi+1)/Math.log(300)*8)));}\n"
 "  else b=mode==='shade'?Math.min(7,Math.floor(esf(i)/100*8)):Math.min(5,Math.floor(Math.log(EFx[i]+1)/lf*6));")
RDW_OLD="else if(colorMode==='plain'){map.getSource('net').setData(binFC('shade',ss));map.setPaintProperty('net','line-color',hi?'#b3bcc4':'#8c8c8c');map.setPaintProperty('net','line-width',LWHALF);}"
RDW_NEW=("else if(colorMode==='rho'){map.getSource('net').setData(binFC('rho',ss));map.setPaintProperty('net','line-color',hi?'#b3bcc4':['match',['get','b'],0,RHOC[0],1,RHOC[1],2,RHOC[2],3,RHOC[3],4,RHOC[4],5,RHOC[5],6,RHOC[6],7,RHOC[7],'#888']);map.setPaintProperty('net','line-width',hi?LWHALF:LWFULL);}\n"
 " else if(colorMode==='plain'){map.getSource('net').setData(binFC('shade',ss));map.setPaintProperty('net','line-color',hi?'#b3bcc4':'#8c8c8c');map.setPaintProperty('net','line-width',LWHALF);}")
SHC_OLD="const SHC=['#b2182b','#d6604d','#f4a582','#fddbc7','#d1e5f0','#92c5de','#4393c3','#2166ac'];"
SHC_NEW=("const SHC=['#b2182b','#d6604d','#f4a582','#fddbc7','#d1e5f0','#92c5de','#4393c3','#2166ac'];\n"
 "const RHOC=['#f7f4ea','#e8dcc0','#d9be92','#c99d68','#b47b47','#96592f','#70401f','#452712'];  // street resistance rho: light=low(shaded), dark=high(sun)")
ULG_OLD="function updLeg(){let l=document.getElementById('leg');if(colorMode==='shade'){"
ULG_NEW=("function updLeg(){let l=document.getElementById('leg');\n"
 " if(colorMode==='rho'){let _bar='';for(let b=0;b<8;b++)_bar+='<span style=\"width:15px;height:10px;display:inline-block;background:'+RHOC[b]+'\"></span>';\n"
 "  l.innerHTML='<b style=\"font-weight:500\">路段成本 ωᵢ=ℓ×ρ(λ='+PLAM+')</b><div style=\"display:flex;margin-top:3px\">'+_bar+'</div>'\n"
 "   +'<div style=\"display:flex;justify-content:space-between;width:120px;font-size:9px;color:#555;margin-top:1px\"><span>低</span><span>ωᵢ = ℓ×ρ</span><span>高</span></div>'\n"
 "   +'<div style=\"font-size:9px;color:#777;margin-top:2px\">最遮荫路 = min ω=Σωᵢ</div>';\n"
 "  l.innerHTML+='<br><span class=\"sw\" style=\"background:#0F9E75\"></span>最遮荫路 &nbsp;<span class=\"sw\" style=\"background:#E0791E\"></span>最短路';return;}\n"
 " if(colorMode==='shade'){")
PATCHES=[(SEC_OLD,SEC_NEW),(SLD_OLD,SLD_NEW),(DEC_OLD,DEC_NEW),(RT_OLD,RT_NEW),(DTH_OLD,DTH_NEW),(LEG_OLD,LEG_NEW),(FC_OLD,FC_NEW),
 (BTN_OLD,BTN_NEW),(HND_OLD,HND_NEW),(BIN_OLD,BIN_NEW),(BINB_OLD,BINB_NEW),(RDW_OLD,RDW_NEW),(SHC_OLD,SHC_NEW),(ULG_OLD,ULG_NEW),
 (TOPOW_OLD,TOPOW_NEW),(TOPOMET_OLD,TOPOMET_NEW),(TOPOST_OLD,TOPOST_NEW),
 (G1TIP_OLD,G1TIP_NEW),(G1LBL_OLD,G1LBL_NEW),(G2TIP_OLD,G2TIP_NEW),(CARD_OLD,CARD_NEW),
 (T1_OLD,T1_NEW),(T2_OLD,T2_NEW),(MET_OLD,MET_NEW),(FOOT_OLD,FOOT_NEW),(HDR_OLD,HDR_NEW),(TITLE_OLD,TITLE_NEW)]
for i,(old,new) in enumerate(PATCHES):
    n=h.count(old)
    assert n==1,("patch %d 匹配 %d 次: "%(i,n))+old[:70]
    h=h.replace(old,new)
# add cRho to the pick arrays of the other 3 Network view handlers (the cPlain one was already changed by the HND patch)
n=h.count(PICK_OLD)
assert n==3,"pick 数组剩余 %d 处(应 3)"%n
h=h.replace(PICK_OLD,PICK_NEW)
open(f"{WEB}\\nav_app_paper.html","w",encoding="utf-8").write(h)
print("已写 nav_app_paper.html,13 处替换全部命中")