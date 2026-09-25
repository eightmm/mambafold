#!/usr/bin/env python
# ruff: noqa: E501
"""Render a dependency-free interactive pLDDT C-alpha viewer from a PDB."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def read_ca_trace(path: Path) -> tuple[list[list[float]], list[float]]:
    coords: list[list[float]] = []
    scores: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        scores.append(float(line[60:66]))
    if len(coords) < 2:
        raise ValueError(f"PDB requires at least two CA atoms: {path}")
    if any(not 0.0 <= score <= 100.0 for score in scores):
        raise ValueError("PDB CA B-factors must contain pLDDT values in [0,100]")
    return coords, scores


def render_fragment(path: Path) -> str:
    coords, scores = read_ca_trace(path)
    payload = json.dumps(
        {"coords": coords, "scores": scores},
        separators=(",", ":"),
        allow_nan=False,
    )
    title = html.escape(path.stem)
    mean_score = sum(scores) / len(scores)
    template = r"""<div id="mf-self-contained-plddt">
  <h2>__TITLE__ · pLDDT-colored Cα trace</h2>
  <div class="text-small text-muted">__N__ residues · mean pLDDT __MEAN__ · drag to rotate · wheel to zoom</div>
  <canvas id="mf-plddt-canvas" aria-label="Interactive protein C-alpha trace colored by pLDDT"></canvas>
  <div class="mf-plddt-legend" aria-label="pLDDT confidence legend">
    <span><i class="mf-vlow"></i>&lt;50 very low</span><span><i class="mf-low"></i>50–70 low</span>
    <span><i class="mf-good"></i>70–90 confident</span><span><i class="mf-high"></i>≥90 very high</span>
  </div>
</div>
<style>
#mf-self-contained-plddt{width:100%}#mf-self-contained-plddt h2{margin-bottom:.3rem}
#mf-plddt-canvas{display:block;width:100%;height:520px;margin-top:.7rem;touch-action:none;background:color-mix(in srgb,var(--muted) 24%,transparent)}
.mf-plddt-legend{display:flex;flex-wrap:wrap;gap:.6rem 1rem;margin-top:.55rem;color:var(--foreground)}
.mf-plddt-legend span{display:inline-flex;align-items:center;gap:.35rem}.mf-plddt-legend i{width:.8rem;height:.8rem;display:inline-block}
.mf-vlow{background:#ff7d45}.mf-low{background:#ffdb13}.mf-good{background:#65cbf3}.mf-high{background:#0053d6}
@media(max-width:520px){#mf-plddt-canvas{height:380px}}
</style>
<script type="application/json" id="mf-self-contained-data">__PAYLOAD__</script>
<script>
(()=>{const root=document.getElementById('mf-self-contained-plddt'),canvas=document.getElementById('mf-plddt-canvas'),ctx=canvas.getContext('2d'),data=JSON.parse(document.getElementById('mf-self-contained-data').textContent);let rx=-.55,ry=.65,zoom=1,drag=false,px=0,py=0;
const color=v=>v<50?'#ff7d45':v<70?'#ffdb13':v<90?'#65cbf3':'#0053d6';
const center=[0,1,2].map(k=>data.coords.reduce((a,p)=>a+p[k],0)/data.coords.length);const pts=data.coords.map(p=>p.map((v,k)=>v-center[k]));
function rotate(p){const cy=Math.cos(ry),sy=Math.sin(ry),cx=Math.cos(rx),sx=Math.sin(rx),x=cy*p[0]+sy*p[2],z=-sy*p[0]+cy*p[2];return[x,cx*p[1]-sx*z,sx*p[1]+cx*z]}
function draw(){const d=devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;if(canvas.width!==Math.round(w*d)||canvas.height!==Math.round(h*d)){canvas.width=Math.round(w*d);canvas.height=Math.round(h*d)}ctx.setTransform(d,0,0,d,0,0);ctx.clearRect(0,0,w,h);const r=pts.map(rotate),span=Math.max(...r.flatMap(p=>[Math.abs(p[0]),Math.abs(p[1])]),1),s=.43*Math.min(w,h)/span*zoom,q=r.map(p=>[w/2+p[0]*s,h/2-p[1]*s,p[2]]);const seg=q.slice(0,-1).map((p,i)=>({a:p,b:q[i+1],i,z:(p[2]+q[i+1][2])/2})).sort((a,b)=>a.z-b.z);ctx.lineCap='round';for(const e of seg){ctx.beginPath();ctx.moveTo(e.a[0],e.a[1]);ctx.lineTo(e.b[0],e.b[1]);ctx.strokeStyle=color((data.scores[e.i]+data.scores[e.i+1])/2);ctx.lineWidth=4;ctx.stroke()}for(const [i,p] of q.entries()){ctx.beginPath();ctx.arc(p[0],p[1],3.2,0,Math.PI*2);ctx.fillStyle=color(data.scores[i]);ctx.fill()} }
canvas.addEventListener('pointerdown',e=>{drag=true;px=e.clientX;py=e.clientY;canvas.setPointerCapture(e.pointerId)});canvas.addEventListener('pointermove',e=>{if(!drag)return;ry+=(e.clientX-px)*.008;rx+=(e.clientY-py)*.008;px=e.clientX;py=e.clientY;draw()});canvas.addEventListener('pointerup',()=>drag=false);canvas.addEventListener('pointercancel',()=>drag=false);canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.35,Math.min(4,zoom*Math.exp(-e.deltaY*.001)));draw()},{passive:false});new ResizeObserver(draw).observe(canvas);draw()})();
</script>
"""
    return (
        template.replace("__TITLE__", title)
        .replace("__N__", str(len(coords)))
        .replace("__MEAN__", f"{mean_score:.1f}")
        .replace("__PAYLOAD__", payload)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fragment = render_fragment(args.pdb)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(fragment, encoding="utf-8")
    print(f"wrote {args.out} ({len(fragment)} bytes)")


if __name__ == "__main__":
    main()
