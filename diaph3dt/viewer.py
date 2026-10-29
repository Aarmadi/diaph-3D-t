"""Interactive HTML viewer of the dynamic reconstruction.

The template and the control point trajectories are embedded in the page; the deformation
of each frame (Gaussian RBF, RBF + affine, thin-plate spline) is computed in the browser,
so the kernel scale and the regularisation can be changed live. The biharmonic operator
is precomputed in Python and embedded."""

import base64
import json
import math
import os
from collections import Counter

import numpy as np
from scipy.spatial import cKDTree

from .deformation import BiharmonicDeformer, closest_barycentric
from .template import load_mesh, mesh_signed_volume

# Number of control points for which the base regularisation is calibrated: the effective
# regularisation grows with n / RBF_REG_N_REF (and with sigma), as the kernel matrix gets
# ill-conditioned with many close points (same rule in the viewer and the evaluation).
RBF_REG_N_REF = 450


def _parse_frames(raw_frames, n_points):
    """JSON frames -> list of dicts with 'points' as an (N, 3) array (NaN = unobserved)."""
    frames = []
    for fr in raw_frames:
        if len(fr['points']) != n_points:
            raise ValueError(f"frame t={fr['t']}: {len(fr['points'])} points, {n_points} expected")
        arr = np.full((n_points, 3), np.nan)
        for i, p in enumerate(fr['points']):
            if p is not None:
                arr[i] = p
        frames.append({
            't': fr['t'], 'points': arr,
            'phi_mean': fr.get('phi_mean'),
            'phi_by_plane': fr.get('phi_by_plane') or {},
            'plane_missing': fr.get('plane_missing') or [],
            'phi_divergent': bool(fr.get('phi_divergent')),
            'interpolated': bool(fr.get('interpolated', False)),
            'origin_plane': fr.get('origin_plane'),
        })
    return frames


def load_points(points_path):
    """Control points written by the reconstruction.
    Returns (rest_points (N, 3), frames, frames_interp or None)."""
    with open(points_path, encoding='utf-8') as f:
        data = json.load(f)
    rest = np.array(data['rest_points'], dtype=np.float64)
    frames = _parse_frames(data['frames'], len(rest))
    frames_interp = (_parse_frames(data['frames_interp'], len(rest))
                     if 'frames_interp' in data else None)
    return rest, frames, frames_interp


def auto_sigma_range(rest_points):
    """(sigma_min, sigma_max, sigma_default) of the Gaussian kernel: from the nearest
    neighbour spacing (local deformation) to half the bounding-box diagonal (global)."""
    if len(rest_points) < 2:
        return 1.0, 10.0, 3.0
    d, _ = cKDTree(rest_points).query(rest_points, k=2)
    sigma_min = max(float(np.percentile(d[:, 1], 10)), 1e-3)
    diag = float(np.linalg.norm(rest_points.max(axis=0) - rest_points.min(axis=0)))
    sigma_max = max(diag / 2, sigma_min * 5)
    return sigma_min, sigma_max, float(np.sqrt(sigma_min * sigma_max))


def _round_verts(verts):
    return [[round(float(x), 3) for x in v] for v in verts]


def _biharm_payload(verts, faces, rest_points, frames, frames_interp, weight):
    """Biharmonic operator G (V x K, float32, base64) for the most frequent set of
    observed points; the page computes G @ d at each frame and falls back to the TPS on
    frames where one of these points is missing. JSON string ('null' if disabled)."""
    if not weight:
        return 'null'
    all_frames = list(frames) + list(frames_interp or [])
    masks = Counter(tuple(bool(v) for v in ~np.isnan(np.asarray(fr['points'])[:, 0]))
                    for fr in all_frames)
    if not masks:
        return 'null'
    idx = np.where(np.array(masks.most_common(1)[0][0]))[0]
    if len(idx) < 4:
        return 'null'
    bf, bw, _ = closest_barycentric(np.asarray(rest_points)[idx], verts, faces)
    G = BiharmonicDeformer(verts, faces).operator(bf, bw, weight).astype(np.float32)
    return json.dumps({'idx': idx.tolist(), 'g': base64.b64encode(G.tobytes()).decode('ascii'),
                       'weight': weight})


def _mesh_payload(mesh):
    if mesh is None:
        return [], [], False
    return _round_verts(mesh[0]), mesh[1].tolist(), mesh_signed_volume(*mesh) < 0


def _frames_to_list(frames):
    out = []
    for fr in frames:
        pts = [None if np.any(np.isnan(row)) else [round(float(c), 3) for c in row]
               for row in fr['points']]
        phi, op = fr.get('phi_mean'), fr.get('origin_plane')
        out.append({
            't': round(float(fr['t']), 4), 'points': pts,
            'phi_mean': round(float(phi), 4) if phi is not None else None,
            'phi_by_plane': fr.get('phi_by_plane') or {},
            'plane_missing': fr.get('plane_missing') or [],
            'phi_divergent': bool(fr.get('phi_divergent')),
            'interpolated': bool(fr.get('interpolated', False)),
            'origin_plane': int(op) if op is not None else None,
        })
    return out


CSS = (
    'body{margin:0;font-family:-apple-system,Segoe UI,sans-serif;'
    'background:#0d1117;color:#cdd9e5}'
    'h1{text-align:center;padding:16px 0 4px;font-size:1.3rem;margin:0}'
    '.sub{text-align:center;color:#888;font-size:.85rem;margin-bottom:10px}'
    '#wrap{display:flex;flex-direction:column;align-items:center;padding:0 16px 24px}'
    'canvas{background:#0d1117;border:1px solid #30363d;border-radius:6px;cursor:grab}'
    'canvas:active{cursor:grabbing}'
    '.ctrl{display:flex;gap:14px;align-items:center;flex-wrap:wrap;'
    'margin-top:10px;max-width:900px;justify-content:center}'
    '.ctrl label{font-size:.82rem;color:#ccc;display:flex;align-items:center;gap:6px}'
    '.ctrl input[type=range]{width:220px;accent-color:#58a6ff}'
    '.ctrl input[type=checkbox]{accent-color:#58a6ff;width:15px;height:15px}'
    '.ctrl button{padding:5px 14px;background:#21262d;border:1px solid #444;'
    'border-radius:4px;color:#ccc;cursor:pointer;font-size:.85rem}'
    '.ctrl button:hover{background:#2d333b}'
    '.info{font-family:monospace;font-size:.8rem;color:#888;margin-top:8px;'
    'text-align:center}'
    '.warn{font-family:monospace;font-size:.8rem;color:#f0b93d;margin-top:4px;'
    'text-align:center;min-height:1.2em}'
    '.leg{font-size:.75rem;color:#666;margin-top:10px;max-width:820px;text-align:center}'
    '#tcv{cursor:pointer}'
)


JS_VIEWER = r"""
var VERTS=__VERTS__, FACES=__FACES__, REST=__REST__, FRAMES=__FRAMES__;
var FRAMES_INTERP=__FRAMES_INTERP__;
var ACTIVE_FRAMES=FRAMES;
var useInterp=false;
var FRAME_INTERVAL_SEC=__FRAME_INTERVAL_SEC__;
var SIGMA_MIN=__SIGMA_MIN__, SIGMA_MAX=__SIGMA_MAX__, SIGMA_DEFAULT=__SIGMA_DEFAULT__;
var REG=__REG__;
var curReg=REG;
var RBF_REG_N_REF=__RBF_REG_N_REF__;
var BEST_SIGMA=__BEST_SIGMA__;
var FLIP_NORMALS=__FLIP_NORMALS__;
var INSPI_VERTS=__INSPI_VERTS__, INSPI_FACES=__INSPI_FACES__, INSPI_FLIP=__INSPI_FLIP__;
var EXPI_VERTS=__EXPI_VERTS__, EXPI_FACES=__EXPI_FACES__, EXPI_FLIP=__EXPI_FLIP__;
var NORMAL_VERTS=__NORMAL_VERTS__, NORMAL_FACES=__NORMAL_FACES__, NORMAL_FLIP=__NORMAL_FLIP__;
var ARAP_AVAILABLE=__ARAP_AVAILABLE__;
var ARAP_VERTS=__ARAP_VERTS__, ARAP_FACES=__ARAP_FACES__, ARAP_FLIP=__ARAP_FLIP__;
var ARAP_FRAMES_T=__ARAP_FRAMES_T__, ARAP_FRAMES_VERTS=__ARAP_FRAMES_VERTS__;
var ARAP_FRAMES_T_INTERP=__ARAP_FRAMES_T_INTERP__, ARAP_FRAMES_VERTS_INTERP=__ARAP_FRAMES_VERTS_INTERP__;
var ARAP_PARAMS_TXT=__ARAP_PARAMS_TXT__;

function phiCol(phi){
  if(phi===null||phi===undefined)return "rgb(120,120,120)";
  if(phi<=0.2)return "rgb(50,210,90)";
  if(phi>=0.8)return "rgb(220,50,50)";
  var t=(phi-0.2)/0.6;
  return "rgb("+Math.round(220*t)+","+Math.round(210*(1-t))+",0)";
}

var PLANE_COLORS=["#58a6ff","#3ddc72","#f0b93d","#ff6b6b","#c678dd"];
function planeColor(pi){ return PLANE_COLORS[pi % PLANE_COLORS.length]; }

function buildEdges(faces){
  var seen={}, edges=[];
  for(var i=0;i<faces.length;i++){
    var f=faces[i];
    var pairs=[[f[0],f[1]],[f[1],f[2]],[f[2],f[0]]];
    for(var j=0;j<3;j++){
      var a=Math.min(pairs[j][0],pairs[j][1]), b=Math.max(pairs[j][0],pairs[j][1]);
      var key=a+"_"+b;
      if(!seen[key]){seen[key]=true;edges.push([a,b]);}
    }
  }
  return edges;
}
var EDGES=buildEdges(FACES);
var ARAP_EDGES=ARAP_AVAILABLE?buildEdges(ARAP_FACES):[];
var INSPI_EDGES=INSPI_FACES.length?buildEdges(INSPI_FACES):[];
var EXPI_EDGES=EXPI_FACES.length?buildEdges(EXPI_FACES):[];
var NORMAL_EDGES=NORMAL_FACES.length?buildEdges(NORMAL_FACES):[];

function computeTraj(){
  var order=ACTIVE_FRAMES.map(function(fr,fi){return fi;});
  order.sort(function(a,b){return ACTIVE_FRAMES[a].t-ACTIVE_FRAMES[b].t;});
  var traj=[];
  for(var i=0;i<REST.length;i++){
    var pts=[];
    for(var k=0;k<order.length;k++){
      var p=ACTIVE_FRAMES[order[k]].points[i];
      if(p!==null)pts.push(p);
    }
    traj.push(pts);
  }
  return traj;
}
var TRAJ=computeTraj();

var bboxMin=[Infinity,Infinity,Infinity], bboxMax=[-Infinity,-Infinity,-Infinity];
for(var i=0;i<VERTS.length;i++){
  for(var k=0;k<3;k++){
    if(VERTS[i][k]<bboxMin[k])bboxMin[k]=VERTS[i][k];
    if(VERTS[i][k]>bboxMax[k])bboxMax[k]=VERTS[i][k];
  }
}
var CENTER=[(bboxMin[0]+bboxMax[0])/2,(bboxMin[1]+bboxMax[1])/2,(bboxMin[2]+bboxMax[2])/2];
var DIAG=Math.hypot(bboxMax[0]-bboxMin[0],bboxMax[1]-bboxMin[1],bboxMax[2]-bboxMin[2])||1;

function gaussK(d2,sigma){return Math.exp(-d2/(2*sigma*sigma));}

function solveLinear(A,B,n,m){
  var M=[];
  for(var i=0;i<n;i++)M.push(A[i].slice().concat(B[i].slice()));
  for(var col=0;col<n;col++){
    var piv=col,best=Math.abs(M[col][col]);
    for(var r=col+1;r<n;r++){if(Math.abs(M[r][col])>best){best=Math.abs(M[r][col]);piv=r;}}
    if(piv!==col){var tmp=M[col];M[col]=M[piv];M[piv]=tmp;}
    var pv=M[col][col]; if(Math.abs(pv)<1e-12)pv=(pv<0?-1:1)*1e-12;
    for(var c=col;c<n+m;c++)M[col][c]/=pv;
    for(var r=0;r<n;r++){
      if(r===col)continue;
      var factor=M[r][col];
      if(factor===0)continue;
      for(var c=col;c<n+m;c++)M[r][c]-=factor*M[col][c];
    }
  }
  var X=[];
  for(var i=0;i<n;i++)X.push(M[i].slice(n,n+m));
  return X;
}

var method="__DEFAULT_METHOD__";
function setMethod(val){
  method=val;
  var sigmaCtrl=document.getElementById("sigma-ctrl");
  if(sigmaCtrl)sigmaCtrl.style.display=(val==="arap"||val==="tps"||val==="biharm")?"none":"flex";
  var regCtrl=document.getElementById("reg-ctrl");
  if(regCtrl)regCtrl.style.display=(val==="exact"||val==="affine"||val==="gp"||val==="tps")?"flex":"none";
  var uncCtrl=document.getElementById("unc-ctrl");
  if(uncCtrl)uncCtrl.style.display=(val==="gp")?"flex":"none";
  needsRecompute=true; needsRender=true;
}

function affineBasis(p){
  return [1,(p[0]-CENTER[0])/DIAG,(p[1]-CENTER[1])/DIAG,(p[2]-CENTER[2])/DIAG];
}
var GP_LARGE_FACTOR=__GP_LARGE_FACTOR__;
var GP_FINE_AMP=__GP_FINE_AMP__;

var BIHARM=__BIHARM__;
var BIHARM_G=null;
function biharmMatrix(){
  if(BIHARM_G===null&&BIHARM){
    var bin=atob(BIHARM.g), buf=new Uint8Array(bin.length);
    for(var i=0;i<bin.length;i++)buf[i]=bin.charCodeAt(i);
    BIHARM_G=new Float32Array(buf.buffer);
  }
  return BIHARM_G;
}
function computeDeformedBiharm(sigma, frameIdx){
  var frame=ACTIVE_FRAMES[frameIdx];
  var ok=!!BIHARM;
  if(ok){for(var a=0;a<BIHARM.idx.length;a++){if(frame.points[BIHARM.idx[a]]===null){ok=false;break;}}}
  if(!ok){
    var fb=computeDeformedExact(sigma, frameIdx, true, false, true);
    fb.biharmFallback=true; return fb;
  }
  var G=biharmMatrix(), K=BIHARM.idx.length, D=new Array(K);
  for(var a=0;a<K;a++){
    var i=BIHARM.idx[a], p=REST[i], q=frame.points[i];
    D[a]=[q[0]-p[0],q[1]-p[1],q[2]-p[2]];
  }
  var out=new Array(VERTS.length);
  for(var vi=0;vi<VERTS.length;vi++){
    var v=VERTS[vi], dx=0,dy=0,dz=0, base=vi*K;
    for(var a=0;a<K;a++){var g=G[base+a]; dx+=g*D[a][0]; dy+=g*D[a][1]; dz+=g*D[a][2];}
    out[vi]=[v[0]+dx,v[1]+dy,v[2]+dz];
  }
  return {verts:out, nValid:K, nTotal:REST.length};
}
var gpStdCache={key:null, std:null};
function kernelK(d2,sigma,gp,tps){
  if(tps)return -Math.sqrt(d2);
  return gp ? GP_FINE_AMP*gaussK(d2,sigma)+gaussK(d2,GP_LARGE_FACTOR*sigma) : gaussK(d2,sigma);
}
function gpPosteriorStd(Mmat, validIdx, sigma, m){
  var Minv=[], I=[];
  for(var r=0;r<m;r++){var e=new Array(m).fill(0); e[r]=1; I.push(e);}
  Minv=solveLinear(Mmat,I,m,m);
  var n=validIdx.length, prior=GP_FINE_AMP+1.0, std=new Array(VERTS.length);
  for(var vi=0;vi<VERTS.length;vi++){
    var v=VERTS[vi], f=new Array(m);
    for(var a=0;a<n;a++){
      var pa=REST[validIdx[a]];
      var dx=v[0]-pa[0],dy=v[1]-pa[1],dz=v[2]-pa[2];
      f[a]=kernelK(dx*dx+dy*dy+dz*dz,sigma,true);
    }
    var bv=affineBasis(v); for(var c=0;c<4;c++)f[n+c]=bv[c];
    var q=0;
    for(var r=0;r<m;r++){var s=0, row=Minv[r]; for(var c=0;c<m;c++)s+=row[c]*f[c]; q+=f[r]*s;}
    std[vi]=Math.sqrt(Math.max(prior-q,0)/prior);
  }
  var lo=Infinity, hi=-Infinity;
  for(var vi=0;vi<std.length;vi++){if(std[vi]<lo)lo=std[vi]; if(std[vi]>hi)hi=std[vi];}
  std.lo=lo; std.hi=hi;
  return std;
}
function computeDeformedExact(sigma, frameIdx, affine, gp, tps){
  var frame=ACTIVE_FRAMES[frameIdx];
  var validIdx=[];
  for(var i=0;i<REST.length;i++){if(frame.points[i]!==null)validIdx.push(i);}
  var n=validIdx.length;
  if(n===0)return {verts:VERTS, nValid:0, nTotal:REST.length};
  if(gp||tps)affine=true;
  if(affine&&n<4){affine=false; gp=false; tps=false;}
  var regEff=curReg*Math.max(1, Math.pow(sigma/SIGMA_DEFAULT, 3))*Math.max(1, n/RBF_REG_N_REF);
  if(tps){
    var lo=[Infinity,Infinity,Infinity], hi=[-Infinity,-Infinity,-Infinity];
    for(var a=0;a<n;a++){var p=REST[validIdx[a]];
      for(var c=0;c<3;c++){if(p[c]<lo[c])lo[c]=p[c]; if(p[c]>hi[c])hi[c]=p[c];}}
    regEff=curReg*Math.hypot(hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2]);
  }
  var m=affine?n+4:n;
  var K=[],D=[];
  for(var a=0;a<n;a++){
    var pa=REST[validIdx[a]], row=[];
    for(var b=0;b<n;b++){
      var pb=REST[validIdx[b]];
      var dx=pa[0]-pb[0],dy=pa[1]-pb[1],dz=pa[2]-pb[2];
      var kv=kernelK(dx*dx+dy*dy+dz*dz,sigma,gp,tps);
      if(a===b)kv+=regEff;
      row.push(kv);
    }
    if(affine)row=row.concat(affineBasis(pa));
    K.push(row);
    var qa=frame.points[validIdx[a]];
    D.push([qa[0]-pa[0],qa[1]-pa[1],qa[2]-pa[2]]);
  }
  if(affine){
    for(var r=0;r<4;r++){
      var row2=[];
      for(var a=0;a<n;a++)row2.push(affineBasis(REST[validIdx[a]])[r]);
      for(var c=0;c<4;c++)row2.push(r===c?-1e-8:0);
      K.push(row2); D.push([0,0,0]);
    }
  }
  var std=null;
  if(gp){
    var key=sigma+"|"+regEff+"|"+validIdx.join(",");
    if(gpStdCache.key!==key){gpStdCache={key:key, std:gpPosteriorStd(K,validIdx,sigma,m)};}
    std=gpStdCache.std;
  }
  var W=solveLinear(K,D,m,3);
  var out=new Array(VERTS.length);
  for(var vi=0;vi<VERTS.length;vi++){
    var v=VERTS[vi], dxv=0,dyv=0,dzv=0;
    for(var a=0;a<n;a++){
      var pa=REST[validIdx[a]];
      var ddx=v[0]-pa[0],ddy=v[1]-pa[1],ddz=v[2]-pa[2];
      var kv=kernelK(ddx*ddx+ddy*ddy+ddz*ddz,sigma,gp,tps);
      dxv+=W[a][0]*kv; dyv+=W[a][1]*kv; dzv+=W[a][2]*kv;
    }
    if(affine){
      var bv=affineBasis(v);
      for(var c=0;c<4;c++){dxv+=W[n+c][0]*bv[c]; dyv+=W[n+c][1]*bv[c]; dzv+=W[n+c][2]*bv[c];}
    }
    out[vi]=[v[0]+dxv,v[1]+dyv,v[2]+dzv];
  }
  return {verts:out, nValid:n, nTotal:REST.length, regEff:regEff, std:std};
}

function computeDeformedShepard(sigma, frameIdx){
  var frame=ACTIVE_FRAMES[frameIdx];
  var validIdx=[];
  for(var i=0;i<REST.length;i++){if(frame.points[i]!==null)validIdx.push(i);}
  var n=validIdx.length;
  if(n===0)return {verts:VERTS, nValid:0, nTotal:REST.length};
  var D=[];
  for(var a=0;a<n;a++){
    var pa=REST[validIdx[a]];
    var qa=frame.points[validIdx[a]];
    D.push([qa[0]-pa[0],qa[1]-pa[1],qa[2]-pa[2]]);
  }
  var out=new Array(VERTS.length);
  for(var vi=0;vi<VERTS.length;vi++){
    var v=VERTS[vi], dxv=0,dyv=0,dzv=0,wsum=0;
    for(var a=0;a<n;a++){
      var pa=REST[validIdx[a]];
      var ddx=v[0]-pa[0],ddy=v[1]-pa[1],ddz=v[2]-pa[2];
      var kv=gaussK(ddx*ddx+ddy*ddy+ddz*ddz,sigma);
      dxv+=D[a][0]*kv; dyv+=D[a][1]*kv; dzv+=D[a][2]*kv;
      wsum+=kv;
    }
    if(wsum<1e-12)wsum=1e-12;
    out[vi]=[v[0]+dxv/wsum,v[1]+dyv/wsum,v[2]+dzv/wsum];
  }
  return {verts:out, nValid:n, nTotal:REST.length};
}

function arapNearestIdxForT(tsArray, t){
  if(!tsArray.length)return -1;
  var bestIdx=0, bestDiff=Infinity;
  for(var i=0;i<tsArray.length;i++){
    var d=Math.abs(tsArray[i]-t);
    if(d<bestDiff){bestDiff=d;bestIdx=i;}
    if(d===0)break;
  }
  return bestIdx;
}

function computeDeformed(sigma, frameIdx){
  if(method==="arap"){
    var arapT=useInterp?ARAP_FRAMES_T_INTERP:ARAP_FRAMES_T;
    var arapVerts=useInterp?ARAP_FRAMES_VERTS_INTERP:ARAP_FRAMES_VERTS;
    if(!ARAP_AVAILABLE||!arapVerts.length){
      return {verts:VERTS, nValid:0, nTotal:REST.length, isArap:true, arapUnavailable:true};
    }
    var idx=arapNearestIdxForT(arapT, ACTIVE_FRAMES[frameIdx].t);
    return {verts:arapVerts[idx], isArap:true, arapT:arapT[idx]};
  }
  if(method==="shepard")return computeDeformedShepard(sigma, frameIdx);
  if(method==="biharm")return computeDeformedBiharm(sigma, frameIdx);
  return computeDeformedExact(sigma, frameIdx, method==="affine", method==="gp", method==="tps");
}

var canvas, ctx, WPX=820, HPX=620;
var tcanvas, tctx;
var rotY=0.6, rotX=-0.35, rotZ=0.0, zoom=1.0, dragging=false, lastX=0, lastY=0;
var curFrame=0, curSigma=SIGMA_DEFAULT, playing=false, timer=null;
var deformed=null, needsRecompute=true, needsRender=true;
var trailDeformed=[];

var autoRot={y:false, x:false, z:false};
var autoRotTimer=null;
var AUTO_ROT_STEP=0.015;
var AXIS_LABELS={y:'Horizontal', x:'Vertical', z:'Roll'};
function toggleAutoRot(axis){
  autoRot[axis]=!autoRot[axis];
  var btn=document.getElementById('rot-play-'+axis);
  if(btn)btn.textContent=(autoRot[axis]?'⏸ ':'▶ ')+AXIS_LABELS[axis];
  if((autoRot.y||autoRot.x||autoRot.z)&&!autoRotTimer){
    autoRotTimer=setInterval(function(){
      var twoPi=2*Math.PI;
      if(autoRot.y){rotY=(rotY+AUTO_ROT_STEP)%twoPi;}
      if(autoRot.x){rotX=(rotX+AUTO_ROT_STEP)%twoPi;}
      if(autoRot.z){rotZ=(rotZ+AUTO_ROT_STEP)%twoPi;}
      needsRender=true;
    },30);
  }else if(!(autoRot.y||autoRot.x||autoRot.z)&&autoRotTimer){
    clearInterval(autoRotTimer);autoRotTimer=null;
  }
}

var mediaRecorder=null, recordedChunks=[];
function startRecording(){
  if(mediaRecorder)return;
  if(!canvas.captureStream){
    alert("Video recording not supported by this browser (no captureStream).");
    return;
  }
  var stream=canvas.captureStream(30);
  var mimeCandidates=['video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm'];
  var mimeType='';
  for(var i=0;i<mimeCandidates.length;i++){
    if(window.MediaRecorder&&MediaRecorder.isTypeSupported(mimeCandidates[i])){mimeType=mimeCandidates[i];break;}
  }
  try{
    mediaRecorder=mimeType?new MediaRecorder(stream,{mimeType:mimeType}):new MediaRecorder(stream);
  }catch(e){
    alert("Video recording not supported by this browser.");
    return;
  }
  recordedChunks=[];
  mediaRecorder.ondataavailable=function(e){if(e.data&&e.data.size>0)recordedChunks.push(e.data);};
  mediaRecorder.onstop=function(){
    var outType=mediaRecorder.mimeType||'video/webm';
    var blob=new Blob(recordedChunks,{type:outType});
    var ext=outType.indexOf('mp4')!==-1?'mp4':'webm';
    var url=URL.createObjectURL(blob);
    var a=document.createElement('a');
    a.href=url; a.download='capture_mesh_'+Date.now()+'.'+ext;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    setTimeout(function(){URL.revokeObjectURL(url);},2000);
    mediaRecorder=null;
  };
  mediaRecorder.start();
  var btnStart=document.getElementById('rec-start'), btnStop=document.getElementById('rec-stop');
  if(btnStart)btnStart.disabled=true;
  if(btnStop)btnStop.disabled=false;
  var ind=document.getElementById('rec-indicator'); if(ind)ind.style.display='inline';
}
function stopRecording(){
  if(!mediaRecorder)return;
  mediaRecorder.stop();
  var btnStart=document.getElementById('rec-start'), btnStop=document.getElementById('rec-stop');
  if(btnStart)btnStart.disabled=false;
  if(btnStop)btnStop.disabled=true;
  var ind=document.getElementById('rec-indicator'); if(ind)ind.style.display='none';
}

var showSurface=false, showWire=true, showValidPts=true, showLostPts=true;
var showInspi=false, showExpi=false, showNormal=false, showAxes=false;
var showTraj=false;
var overlayWire=true;
function toggleLayer(name, val){
  if(name==="surface")showSurface=val;
  else if(name==="wire")showWire=val;
  else if(name==="validpts")showValidPts=val;
  else if(name==="lostpts")showLostPts=val;
  else if(name==="inspi")showInspi=val;
  else if(name==="expi")showExpi=val;
  else if(name==="normal")showNormal=val;
  else if(name==="axes")showAxes=val;
  else if(name==="traj")showTraj=val;
  else if(name==="overlaywire")overlayWire=val;
  needsRender=true;
}
var trailCount=0;
function setTrail(val){
  trailCount=parseInt(val,10)||0;
  needsRecompute=true; needsRender=true;
}

function camSpace(pt){
  var x=pt[0]-CENTER[0], y=pt[1]-CENTER[1], z=pt[2]-CENTER[2];
  var cz=Math.cos(rotZ), sz=Math.sin(rotZ);
  var xa=x*cz - y*sz, ya=x*sz + y*cz, za=z;
  var cy=Math.cos(rotY), sy=Math.sin(rotY);
  var xb=xa*cy - za*sy, zb=xa*sy + za*cy, yb=ya;
  var cx=Math.cos(rotX), sx=Math.sin(rotX);
  var yc=yb*cx - zb*sx, zc=yb*sx + zb*cx;
  return [xb,yc,zc];
}
function projectCam(cam){
  var scale=(Math.min(WPX,HPX)*0.38*zoom)/DIAG;
  var f=3.2*DIAG;
  var depth=f+cam[2];
  if(depth<1)depth=1;
  var persp=f/depth;
  return [WPX/2 + cam[0]*scale*persp, HPX/2 - cam[1]*scale*persp];
}
function project(pt){ return projectCam(camSpace(pt)); }

function sub3(a,b){ return [a[0]-b[0],a[1]-b[1],a[2]-b[2]]; }
function cross3(a,b){
  return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
}

var SURF_R=74, SURF_G=133, SURF_B=188;
var AMBIENT=0.35, DIFFUSE=0.65;
var TRAIL_ALPHAS=[0.45,0.30,0.18];
var TRAIL_RGB=[[170,110,255],[150,120,235],[130,130,215]];

function uncColor(s){
  var t=Math.max(0,Math.min(1,s));
  return [Math.round(40+215*t), Math.round(120*(1-Math.abs(2*t-1))+60), Math.round(230*(1-t)+25)];
}
var showUnc=false;
function setUncToggle(on){
  showUnc=on;
  if(on&&!showSurface){
    showSurface=true;
    var cb=document.getElementById("surface-cb"); if(cb)cb.checked=true;
  }
  needsRender=true;
}
function drawMeshSurface(meshVerts, meshFaces, flipNormals, baseR, baseG, baseB, alpha, vertStd){
  var n=meshVerts.length;
  var cams=new Array(n), proj=new Array(n);
  for(var i=0;i<n;i++){ cams[i]=camSpace(meshVerts[i]); proj[i]=projectCam(cams[i]); }
  var f_val=3.2*DIAG;
  var camEye=[0,0,-f_val];
  var visible=[];
  for(var fi=0;fi<meshFaces.length;fi++){
    var face=meshFaces[fi];
    var ca=cams[face[0]], cb=cams[face[1]], cc=cams[face[2]];
    var nrm=cross3(sub3(cb,ca), sub3(cc,ca));
    if(flipNormals){nrm=[-nrm[0],-nrm[1],-nrm[2]];}
    var cx=(ca[0]+cb[0]+cc[0])/3, cy=(ca[1]+cb[1]+cc[1])/3, cz=(ca[2]+cb[2]+cc[2])/3;
    var viewDir=[cx-camEye[0], cy-camEye[1], cz-camEye[2]];
    var dot=nrm[0]*viewDir[0]+nrm[1]*viewDir[1]+nrm[2]*viewDir[2];
    if(dot>=0)continue;
    visible.push({fi:fi, depth:cz, nrm:nrm});
  }
  visible.sort(function(a,b){ return b.depth-a.depth; });
  for(var k=0;k<visible.length;k++){
    var item=visible[k], face=meshFaces[item.fi], nrm=item.nrm;
    var nlen=Math.hypot(nrm[0],nrm[1],nrm[2])||1;
    var shade=AMBIENT+DIFFUSE*Math.abs(nrm[2]/nlen);
    if(shade>1)shade=1;
    var cr=baseR, cg=baseG, cb=baseB;
    if(vertStd){
      var s=((vertStd[face[0]]+vertStd[face[1]]+vertStd[face[2]])/3-vertStd.lo)/
            Math.max(vertStd.hi-vertStd.lo,1e-9);
      var col=uncColor(s);
      cr=col[0]; cg=col[1]; cb=col[2];
    }
    ctx.fillStyle="rgba("+Math.round(cr*shade)+","+Math.round(cg*shade)+","+Math.round(cb*shade)+","+alpha+")";
    var pa=proj[face[0]], pb=proj[face[1]], pc=proj[face[2]];
    ctx.beginPath();
    ctx.moveTo(pa[0],pa[1]); ctx.lineTo(pb[0],pb[1]); ctx.lineTo(pc[0],pc[1]);
    ctx.closePath(); ctx.fill();
  }
}

function drawMeshWire(meshVerts, meshEdges, color){
  var proj=new Array(meshVerts.length);
  for(var i=0;i<meshVerts.length;i++)proj[i]=project(meshVerts[i]);
  ctx.strokeStyle=color; ctx.lineWidth=0.8;
  ctx.beginPath();
  for(var e=0;e<meshEdges.length;e++){
    var a=proj[meshEdges[e][0]], b=proj[meshEdges[e][1]];
    ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]);
  }
  ctx.stroke();
}

function drawAxes(){
  var len=DIAG*0.5;
  var p0=project(bboxMin);
  [
    {end:[bboxMin[0]+len,bboxMin[1],bboxMin[2]], color:"#ff6b6b", label:"R"},
    {end:[bboxMin[0],bboxMin[1]+len,bboxMin[2]], color:"#6bff8f", label:"A"},
    {end:[bboxMin[0],bboxMin[1],bboxMin[2]+len], color:"#6ba8ff", label:"S"}
  ].forEach(function(ax){
    var p1=project(ax.end);
    ctx.strokeStyle=ax.color; ctx.lineWidth=2;
    ctx.beginPath(); ctx.moveTo(p0[0],p0[1]); ctx.lineTo(p1[0],p1[1]); ctx.stroke();
    ctx.fillStyle=ax.color; ctx.font="bold 12px monospace";
    ctx.fillText(ax.label, p1[0]+4, p1[1]);
  });
}

function render(){
  if(!deformed)return;
  var grad=ctx.createRadialGradient(WPX/2,HPX/2,0,WPX/2,HPX/2,Math.max(WPX,HPX)*0.75);
  grad.addColorStop(0,"#242c3a"); grad.addColorStop(1,"#0c1018");
  ctx.fillStyle=grad; ctx.fillRect(0,0,WPX,HPX);

  if(!overlayWire){
    if(showInspi&&INSPI_VERTS.length){
      drawMeshSurface(INSPI_VERTS,INSPI_FACES,INSPI_FLIP,90,200,120,0.30);
    }
    if(showExpi&&EXPI_VERTS.length){
      drawMeshSurface(EXPI_VERTS,EXPI_FACES,EXPI_FLIP,210,90,90,0.30);
    }
    if(showNormal&&NORMAL_VERTS.length){
      drawMeshSurface(NORMAL_VERTS,NORMAL_FACES,NORMAL_FLIP,230,200,70,0.30);
    }
  }
  var activeFaces=(method==="arap")?ARAP_FACES:FACES;
  var activeFlip=(method==="arap")?ARAP_FLIP:FLIP_NORMALS;
  var activeEdges=(method==="arap")?ARAP_EDGES:EDGES;

  for(var ti=trailDeformed.length-1; ti>=0; ti--){
    var tr=trailDeformed[ti], c=TRAIL_RGB[ti]||TRAIL_RGB[TRAIL_RGB.length-1];
    var a=TRAIL_ALPHAS[ti]!==undefined?TRAIL_ALPHAS[ti]:0.08;
    drawMeshSurface(tr.verts, activeFaces, activeFlip, c[0], c[1], c[2], a);
    drawMeshWire(tr.verts, activeEdges, "rgba("+c[0]+","+c[1]+","+c[2]+","+(a*0.9).toFixed(2)+")");
  }

  var n=deformed.verts.length;
  var cams=new Array(n), proj=new Array(n);
  for(var i=0;i<n;i++){ cams[i]=camSpace(deformed.verts[i]); proj[i]=projectCam(cams[i]); }

  if(showSurface){
    drawMeshSurface(deformed.verts, activeFaces, activeFlip, SURF_R, SURF_G, SURF_B, 1,
                    (method==="gp"&&showUnc&&deformed.std)?deformed.std:null);
  }

  if(showWire){
    ctx.strokeStyle=showSurface?"rgba(5,8,12,0.35)":"rgba(88,166,255,0.5)";
    ctx.lineWidth=1;
    ctx.beginPath();
    for(var e=0;e<activeEdges.length;e++){
      var ed=activeEdges[e], a=proj[ed[0]], b=proj[ed[1]];
      ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]);
    }
    ctx.stroke();
  }

  if(overlayWire){
    if(showInspi&&INSPI_VERTS.length)drawMeshWire(INSPI_VERTS,INSPI_EDGES,"rgba(90,220,120,0.55)");
    if(showExpi&&EXPI_VERTS.length)drawMeshWire(EXPI_VERTS,EXPI_EDGES,"rgba(255,95,95,0.6)");
    if(showNormal&&NORMAL_VERTS.length)drawMeshWire(NORMAL_VERTS,NORMAL_EDGES,"rgba(255,215,70,0.6)");
  }

  if(showTraj){
    ctx.strokeStyle="rgba(255,190,60,0.35)";
    ctx.lineWidth=1;
    ctx.beginPath();
    for(var ti2=0;ti2<TRAJ.length;ti2++){
      var tr=TRAJ[ti2];
      if(tr.length<2)continue;
      var p0=project(tr[0]);
      ctx.moveTo(p0[0],p0[1]);
      for(var tk=1;tk<tr.length;tk++){
        var pk=project(tr[tk]);
        ctx.lineTo(pk[0],pk[1]);
      }
    }
    ctx.stroke();
  }

  var frame=ACTIVE_FRAMES[curFrame];
  if(showValidPts||showLostPts){
    for(var i=0;i<REST.length;i++){
      var p=frame.points[i];
      var valid=(p!==null);
      if(valid && !showValidPts)continue;
      if(!valid && !showLostPts)continue;
      var pos=valid?p:REST[i];
      var pr=project(pos);
      ctx.beginPath();
      ctx.arc(pr[0],pr[1],valid?3.5:2.2,0,6.283);
      ctx.fillStyle=valid?"#3ddc72":"rgba(255,90,90,0.75)";
      ctx.fill();
      ctx.strokeStyle="rgba(0,0,0,0.55)"; ctx.lineWidth=1; ctx.stroke();
    }
  }

  if(showAxes)drawAxes();

  var interpTxt=frame.interpolated?"   [interpolated]":"";
  var infoEl=document.getElementById("info");
  if(infoEl){
    if(deformed.isArap){
      var arapTTxt=deformed.arapUnavailable?"":
        (deformed.arapT!==frame.t?("   (ARAP shown t="+deformed.arapT+", closest "+
          "disponible)"):"");
      infoEl.textContent="Frame t="+frame.t+interpTxt+"   method=ARAP (precomputed, "+ARAP_PARAMS_TXT+")"+
        arapTTxt+(deformed.arapUnavailable?"   [ARAP unavailable]":"");
    }else{
      var methodLbl=(method==="shepard")?"Shepard/IDW":(method==="affine")?"RBF + affine":
                    (method==="gp")?"GP (2 scales)":(method==="tps")?"TPS":
                    (method==="biharm")?(deformed.biharmFallback?"biharmonic -> TPS (missing points)":
                                         "biharmonic"):"Gaussian RBF";
      var regTxt=(deformed.regEff!==undefined)?("   reg="+deformed.regEff.toExponential(2)):"";
      if(method==="gp"&&showUnc&&deformed.std){
        regTxt+="   uncertainty (relative to the prior): blue "+deformed.std.lo.toFixed(2)+
                " -> red "+deformed.std.hi.toFixed(2);
      }
      infoEl.textContent="Frame t="+frame.t+interpTxt+"   sigma="+curSigma.toFixed(2)+
        "   method="+methodLbl+regTxt+
        "   observed points: "+deformed.nValid+"/"+deformed.nTotal;
    }
  }

  var warnEl=document.getElementById("warn");
  if(warnEl){
    var msgs=[];
    if(frame.plane_missing&&frame.plane_missing.length){
      msgs.push("plane(s) "+frame.plane_missing.join(",")+" without data at this frame");
    }
    if(frame.phi_divergent){
      msgs.push("phase disagreement between planes at this frame (phi_by_plane="+
        JSON.stringify(frame.phi_by_plane)+")");
    }
    warnEl.textContent=msgs.length?("Warning: "+msgs.join("   |   ")):"";
  }

  drawTrace();
}
function drawTrace(){
  if(!tctx)return;
  var tw=tcanvas.width, th=tcanvas.height;
  tctx.clearRect(0,0,tw,th);
  tctx.fillStyle="#111827";tctx.fillRect(0,0,tw,th);
  var mx=40, my=16;
  var phis=ACTIVE_FRAMES.map(function(f){return f.phi_mean;});
  var vals=phis.filter(function(v){return v!==null&&v!==undefined;});
  var vmin=vals.length?Math.min.apply(null,vals):0, vmax=vals.length?Math.max.apply(null,vals):1;
  var pad=(vmax-vmin)*0.1||0.1; vmin-=pad; vmax+=pad;
  var n=ACTIVE_FRAMES.length;
  function xOf(i){return mx+(n>1?i/(n-1):0)*(tw-2*mx);}
  function yOf(phi){return th-my-(phi-vmin)/(vmax-vmin)*(th-2*my);}
  [0,1].forEach(function(v){
    if(v<vmin||v>vmax)return;
    var y=yOf(v);
    tctx.strokeStyle="rgba(255,255,255,0.12)";tctx.lineWidth=1;
    tctx.beginPath();tctx.moveTo(mx,y);tctx.lineTo(tw-mx,y);tctx.stroke();
    tctx.fillStyle="#666";tctx.font="9px monospace";tctx.textAlign="right";
    tctx.fillText(v.toFixed(0),mx-4,y+3);
  });
  tctx.strokeStyle="rgba(88,166,255,0.6)";tctx.lineWidth=1.3;
  tctx.beginPath();
  var started=false;
  for(var i=0;i<n;i++){
    var v=phis[i];
    if(v===null||v===undefined){started=false;continue;}
    var x=xOf(i), y=yOf(v);
    if(!started){tctx.moveTo(x,y);started=true;}else{tctx.lineTo(x,y);}
  }
  tctx.stroke();
  for(var i2=0;i2<n;i2++){
    var v2=phis[i2];
    if(v2===null||v2===undefined)continue;
    var x2=xOf(i2), y2=yOf(v2);
    var flagged=ACTIVE_FRAMES[i2].phi_divergent||(ACTIVE_FRAMES[i2].plane_missing&&ACTIVE_FRAMES[i2].plane_missing.length);
    var origPlane=ACTIVE_FRAMES[i2].origin_plane;
    if(ACTIVE_FRAMES[i2].interpolated){
      tctx.beginPath();tctx.arc(x2,y2,1.6,0,6.283);
      tctx.strokeStyle=phiCol(v2);tctx.lineWidth=1;tctx.stroke();
    }else if(origPlane!==null&&origPlane!==undefined){
      tctx.beginPath();tctx.arc(x2,y2,flagged?3.2:2.6,0,6.283);
      tctx.fillStyle=planeColor(origPlane);tctx.fill();
      if(flagged){tctx.strokeStyle="#fff";tctx.lineWidth=1;tctx.stroke();}
    }else{
      tctx.beginPath();tctx.arc(x2,y2,flagged?3.2:2.2,0,6.283);
      tctx.fillStyle=flagged?"#f0b93d":phiCol(v2);tctx.fill();
    }
  }
  var vc=phis[curFrame];
  if(vc!==null&&vc!==undefined){
    var xc=xOf(curFrame), yc=yOf(vc);
    tctx.beginPath();tctx.arc(xc,yc,6,0,6.283);
    tctx.fillStyle="#ff3b3b";tctx.fill();
    tctx.strokeStyle="#fff";tctx.lineWidth=1.5;tctx.stroke();
  }
}
function clickTrace(evt){
  var r=tcanvas.getBoundingClientRect();
  var mx=(evt.clientX-r.left)*(tcanvas.width/r.width);
  var m=40;
  var n=ACTIVE_FRAMES.length;
  var iFloat=(n>1)?((mx-m)/(tcanvas.width-2*m))*(n-1):0;
  var idx=Math.round(iFloat);
  if(idx<0||idx>=n)return;
  if(filterCorrelated&&visIdx.length){
    idx=visIdx.reduce(function(best,cand){
      return Math.abs(cand-idx)<Math.abs(best-idx)?cand:best;
    },visIdx[0]);
  }
  setCurFrame(idx);
}

var filterCorrelated=false, visIdx=[];
function isCorrelated(i){
  var f=ACTIVE_FRAMES[i];
  return !f.phi_divergent && (!f.plane_missing || f.plane_missing.length===0);
}
function rebuildVisIdx(){
  visIdx=[];
  for(var i=0;i<ACTIVE_FRAMES.length;i++){
    if(!filterCorrelated||isCorrelated(i))visIdx.push(i);
  }
  var sl=document.getElementById("frame-sl");
  if(sl)sl.max=Math.max(0,visIdx.length-1);
}
function toggleFilterCorrelated(val){
  filterCorrelated=val;
  rebuildVisIdx();
  if(!visIdx.length)return;
  if(visIdx.indexOf(curFrame)<0){
    setCurFrame(visIdx.reduce(function(best,cand){
      return Math.abs(cand-curFrame)<Math.abs(best-curFrame)?cand:best;
    },visIdx[0]));
  }else{
    var sl=document.getElementById("frame-sl"); if(sl)sl.value=visIdx.indexOf(curFrame);
  }
}
function setCurFrame(idx){
  curFrame=idx;
  var sl=document.getElementById("frame-sl");
  if(sl){var pos=visIdx.indexOf(idx); sl.value=pos<0?0:pos;}
  var lbl=document.getElementById("frame-val"); if(lbl)lbl.textContent=ACTIVE_FRAMES[idx].t;
  needsRecompute=true; needsRender=true;
}

function setInterpToggle(val){
  var curT=ACTIVE_FRAMES[curFrame]?ACTIVE_FRAMES[curFrame].t:0;
  useInterp=val;
  ACTIVE_FRAMES=val?FRAMES_INTERP:FRAMES;
  TRAJ=computeTraj();
  rebuildVisIdx();
  var pool=visIdx.length?visIdx:ACTIVE_FRAMES.map(function(fr,fi){return fi;});
  var nearest=pool.reduce(function(best,cand){
    return Math.abs(ACTIVE_FRAMES[cand].t-curT)<Math.abs(ACTIVE_FRAMES[best].t-curT)?cand:best;
  },pool[0]);
  setCurFrame(nearest);
  drawTrace();
}

function loop(){
  if(needsRecompute){
    deformed=computeDeformed(curSigma, curFrame);
    trailDeformed=[];
    if(trailCount>0){
      var curT=ACTIVE_FRAMES[curFrame].t;
      var prev=ACTIVE_FRAMES.map(function(fr,fi){return fi;})
        .filter(function(fi){return ACTIVE_FRAMES[fi].t<curT-1e-9;})
        .sort(function(a,b){return ACTIVE_FRAMES[b].t-ACTIVE_FRAMES[a].t;});
      for(var k=0;k<Math.min(trailCount,prev.length);k++)
        trailDeformed.push(computeDeformed(curSigma, prev[k]));
    }
    needsRecompute=false;
  }
  if(needsRender){
    render();
    needsRender=false;
  }
  requestAnimationFrame(loop);
}

function setSigma(val){
  curSigma=parseFloat(val);
  var lbl=document.getElementById("sigma-val"); if(lbl)lbl.textContent=curSigma.toFixed(2);
  needsRecompute=true; needsRender=true;
}
function setReg(val){
  curReg=Math.pow(10, parseFloat(val));
  var lbl=document.getElementById("reg-val"); if(lbl)lbl.textContent=curReg.toExponential(2);
  needsRecompute=true; needsRender=true;
}
function applyPreset(metric){
  if(!BEST_SIGMA||!BEST_SIGMA[metric])return;
  setSigma(BEST_SIGMA[metric].sigma);
  var sl=document.getElementById("sigma-sl"); if(sl)sl.value=curSigma;
}
function seekFrame(posStr){
  var pos=parseInt(posStr,10);
  if(!visIdx.length)return;
  setCurFrame(visIdx[Math.max(0,Math.min(pos,visIdx.length-1))]);
}
function scheduleNextFrame(){
  if(!playing||!visIdx.length)return;
  var pos=visIdx.indexOf(curFrame);
  var nextPos=(pos+1)%visIdx.length;
  var dt=Math.abs(ACTIVE_FRAMES[visIdx[nextPos]].t-ACTIVE_FRAMES[visIdx[pos]].t);
  var delayMs=Math.max(1, dt*FRAME_INTERVAL_SEC*1000);
  timer=setTimeout(function(){
    setCurFrame(visIdx[nextPos]);
    scheduleNextFrame();
  }, delayMs);
}
function togglePlay(){
  if(!visIdx.length)return;
  playing=!playing;
  var btn=document.getElementById("play-btn");
  if(playing){
    if(btn)btn.textContent="Pause";
    scheduleNextFrame();
  }else{
    if(btn)btn.textContent="Play";
    if(timer){clearTimeout(timer);timer=null;}
  }
}

function initViewer(){
  canvas=document.getElementById("cv");
  ctx=canvas.getContext("2d");
  tcanvas=document.getElementById("tcv");
  tctx=tcanvas.getContext("2d");
  tcanvas.addEventListener("click",clickTrace);
  canvas.addEventListener("mousedown",function(e){dragging=true;lastX=e.clientX;lastY=e.clientY;});
  window.addEventListener("mouseup",function(){dragging=false;});
  window.addEventListener("mousemove",function(e){
    if(!dragging)return;
    rotY+=(e.clientX-lastX)*0.01;
    rotX+=(e.clientY-lastY)*0.01;
    rotX=Math.max(-1.5,Math.min(1.5,rotX));
    lastX=e.clientX; lastY=e.clientY;
    needsRender=true;
  });
  canvas.addEventListener("wheel",function(e){
    e.preventDefault();
    zoom*=(e.deltaY<0)?1.08:0.93;
    zoom=Math.max(0.2,Math.min(6,zoom));
    needsRender=true;
  },{passive:false});
  rebuildVisIdx();
  requestAnimationFrame(loop);
}
"""

METHOD_LABELS = {'exact': 'Gaussian RBF', 'affine': 'RBF + affine',
                 'tps': 'Thin-plate spline', 'biharm': 'Biharmonic (on the surface)'}


def build_html(verts, faces, rest_points, frames, title, frames_interp=None,
               inspi_mesh=None, expi_mesh=None, normal_mesh=None, sigma=None, reg=1e-2,
               default_method='tps', biharm_weight=1.0, frame_interval_sec=1.7367):
    """Deformation viewer page.

    verts, faces: template (upper surface of the inspiratory mesh); rest_points (N, 3) and
    frames: control points (see load_points); frames_interp: optional interpolated set,
    toggled live in the page. *_mesh: (verts, faces) of the breath-hold surfaces, shown
    as static references. sigma (mm) and reg: initial Gaussian kernel scale and base
    regularisation. default_method: 'exact' (Gaussian RBF), 'affine', 'tps' or 'biharm'.
    frame_interval_sec: time between two frames of a plane (playback speed only)."""
    sigma_min, sigma_max, sigma_auto = auto_sigma_range(rest_points)
    sigma = sigma_auto if sigma is None else float(sigma)
    sigma_min, sigma_max = min(sigma_min, sigma), max(sigma_max, sigma)
    biharm_json = _biharm_payload(verts, faces, rest_points, frames, frames_interp,
                                  biharm_weight)
    if default_method == 'biharm' and biharm_json == 'null':
        default_method = 'tps'
    inspi = _mesh_payload(inspi_mesh)
    expi = _mesh_payload(expi_mesh)
    normal = _mesh_payload(normal_mesh)
    frames_list = _frames_to_list(frames)
    frames_interp_list = _frames_to_list(frames_interp) if frames_interp is not None else []
    has_origin_plane = any(fr['origin_plane'] is not None for fr in frames_list)

    subs = {
        '__VERTS__': json.dumps(_round_verts(verts)),
        '__FACES__': json.dumps(faces.tolist()),
        '__REST__': json.dumps(_round_verts(rest_points)),
        '__FRAMES__': json.dumps(frames_list, separators=(',', ':')),
        '__FRAMES_INTERP__': json.dumps(frames_interp_list, separators=(',', ':')),
        '__FRAME_INTERVAL_SEC__': repr(float(frame_interval_sec)),
        '__SIGMA_MIN__': repr(sigma_min), '__SIGMA_MAX__': repr(sigma_max),
        '__SIGMA_DEFAULT__': repr(sigma), '__REG__': repr(reg),
        '__RBF_REG_N_REF__': repr(RBF_REG_N_REF), '__BEST_SIGMA__': 'null',
        '__FLIP_NORMALS__': 'true' if mesh_signed_volume(verts, faces) < 0 else 'false',
        '__INSPI_VERTS__': json.dumps(inspi[0]), '__INSPI_FACES__': json.dumps(inspi[1]),
        '__INSPI_FLIP__': 'true' if inspi[2] else 'false',
        '__EXPI_VERTS__': json.dumps(expi[0]), '__EXPI_FACES__': json.dumps(expi[1]),
        '__EXPI_FLIP__': 'true' if expi[2] else 'false',
        '__NORMAL_VERTS__': json.dumps(normal[0]), '__NORMAL_FACES__': json.dumps(normal[1]),
        '__NORMAL_FLIP__': 'true' if normal[2] else 'false',
        # The page script also supports a precomputed (ARAP) sequence and a two-scale
        # Gaussian process; both are disabled here.
        '__ARAP_AVAILABLE__': 'false', '__ARAP_VERTS__': '[]', '__ARAP_FACES__': '[]',
        '__ARAP_FLIP__': 'false', '__ARAP_FRAMES_T__': '[]', '__ARAP_FRAMES_VERTS__': '[]',
        '__ARAP_FRAMES_T_INTERP__': '[]', '__ARAP_FRAMES_VERTS_INTERP__': '[]',
        '__ARAP_PARAMS_TXT__': '""', '__GP_LARGE_FACTOR__': '4.0', '__GP_FINE_AMP__': '0.25',
        '__DEFAULT_METHOD__': default_method, '__BIHARM__': biharm_json,
    }
    js = JS_VIEWER
    for key, value in subs.items():
        js = js.replace(key, value)

    sigma_step = (sigma_max - sigma_min) / 400 if sigma_max > sigma_min else 0.01
    reg_exp = math.log10(reg) if reg > 0 else -6.0
    reg_min, reg_max = min(-9.0, reg_exp - 3.0), max(0.0, reg_exp + 3.0)
    first_t = frames_list[0]['t'] if frames_list else 0
    methods = ['exact', 'affine', 'tps'] + (['biharm'] if biharm_json != 'null' else [])
    method_radios = ''.join(
        f'<label><input type="radio" name="method" value="{m}" '
        f'{"checked" if default_method == m else ""} onchange="setMethod(this.value)"> '
        f'{METHOD_LABELS[m]}</label>' for m in methods)

    def checkbox(layer, label, checked=False, extra=''):
        return (f'<label><input type="checkbox" {extra}{"checked " if checked else ""}'
                f'onchange="toggleLayer(\'{layer}\',this.checked)"> {label}</label>')

    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>Dynamic reconstruction - {title}</title>'
        f'<style>{CSS}</style></head><body onload="initViewer()">'
        f'<h1>Dynamic 3D reconstruction - {title}</h1>'
        f'<p class="sub">{len(verts)} vertices, {len(faces)} faces, '
        f'{len(rest_points)} control points, {len(frames_list)} frames</p>'
        '<div id="wrap">'
        '<canvas id="cv" width="820" height="620"></canvas>'
        '<div class="ctrl">'
        '<span style="color:#888;font-size:.8rem">Auto-rotation:</span>'
        '<button id="rot-play-y" onclick="toggleAutoRot(\'y\')">&#9654; Horizontal</button>'
        '<button id="rot-play-x" onclick="toggleAutoRot(\'x\')">&#9654; Vertical</button>'
        '<button id="rot-play-z" onclick="toggleAutoRot(\'z\')">&#9654; Roll</button>'
        '<button id="rec-start" onclick="startRecording()">&#9210; Record</button>'
        '<button id="rec-stop" onclick="stopRecording()" disabled>&#9209; Stop</button>'
        '<span id="rec-indicator" style="display:none;color:#ff3b3b;font-size:.8rem">'
        '&#9679; REC</span>'
        '</div>'
        '<p style="color:#888;font-size:.75rem;margin:8px 0 4px;text-align:center">'
        'Respiratory trace (phi of each frame, 0 = inspiratory breath-hold, 1 = expiratory '
        'breath-hold); click to select a frame.</p>'
        '<canvas id="tcv" width="820" height="130"></canvas>'
        '<div class="ctrl">'
        '<button id="play-btn" onclick="togglePlay()">Play</button>'
        f'<input type="range" id="frame-sl" min="0" max="{max(0, len(frames_list) - 1)}" '
        'value="0" oninput="seekFrame(this.value)">'
        f'<span>frame t=<span id="frame-val">{first_t}</span></span>'
        + (('<label><input type="checkbox" onchange="setInterpToggle(this.checked)"> '
            f'Intermediate frames ({len(frames_interp_list) - len(frames_list)} '
            'interpolated)</label>') if frames_interp_list else '')
        + '</div>'
        f'<div class="ctrl" id="sigma-ctrl" style="display:'
        f'{"flex" if default_method in ("exact", "affine") else "none"}">'
        f'<label>Gaussian kernel sigma (mm): <span id="sigma-val">{sigma:.2f}</span>'
        f'<input type="range" id="sigma-sl" min="{sigma_min}" max="{sigma_max}" '
        f'step="{sigma_step}" value="{sigma}" oninput="setSigma(this.value)"></label>'
        '</div>'
        f'<div class="ctrl" id="reg-ctrl" style="display:'
        f'{"flex" if default_method in ("exact", "affine", "tps") else "none"}">'
        f'<label>Regularisation: <span id="reg-val">{reg:.2e}</span>'
        f'<input type="range" id="reg-sl" min="{reg_min}" max="{reg_max}" '
        f'step="0.05" value="{reg_exp}" oninput="setReg(this.value)"></label>'
        '</div>'
        f'<div class="ctrl">{method_radios}</div>'
        '<div class="ctrl">'
        + checkbox('surface', 'Shaded surface', extra='id="surface-cb" ')
        + checkbox('wire', 'Wireframe', True)
        + checkbox('validpts', 'Observed points (green)', True)
        + checkbox('lostpts', 'Unobserved points (red)', True)
        + checkbox('axes', 'RAS axes')
        + checkbox('traj', 'Point trajectories')
        + '</div><div class="ctrl">'
        + (checkbox('inspi', 'Inspiratory breath-hold') if inspi[0] else '')
        + (checkbox('expi', 'Expiratory breath-hold') if expi[0] else '')
        + (checkbox('normal', 'Spontaneous breath-hold') if normal[0] else '')
        + (checkbox('overlaywire', 'Breath-holds as wireframe', True)
           if (inspi[0] or expi[0] or normal[0]) else '')
        + '<label>Previous instants: <select onchange="setTrail(this.value)">'
          '<option value="0">0</option><option value="1">1</option>'
          '<option value="2">2</option><option value="3">3</option></select></label>'
        '</div>'
        '<p id="info" class="info"></p>'
        '<p id="warn" class="warn"></p>'
        '<p class="leg">Drag to rotate, wheel to zoom. Green points: control points observed '
        'at this frame (they drive the deformation); red points: not observed at this frame '
        '(shown at their rest position, not used). The breath-hold surfaces are static '
        'references (green = inspiratory, red = expiratory, yellow = spontaneous). '
        'The Gaussian RBF interpolates the observed displacements with a kernel of scale '
        'sigma; RBF + affine adds a global affine motion; the thin-plate spline has no scale; '
        'the biharmonic deformation propagates the displacements along the surface. '
        'The regularisation relaxes the exact interpolation of the points.'
        + (' On the respiratory trace, frames are coloured by their plane of origin: the '
           'planes of a view are acquired in turn, a smooth curve supports the uniform '
           'acquisition delay between planes.' if has_origin_plane else '')
        + '</p></div>'
        f'<script>{js}</script>'
        '</body></html>'
    )


def build_case_html(case_dir, view, title, sigma=None, reg=1e-2, default_method='tps',
                    biharm_weight=1.0, step_size=2):
    """Viewer page of a reconstructed case, from the files written by the reconstruction
    (control_points.json and the filtered breath-hold masks)."""
    from .io import case_file
    verts, faces = load_mesh(case_file(case_dir, 'inspi'), view, step_size=step_size,
                             prefiltered=True, top_only=True)
    rest, frames, frames_interp = load_points(case_file(case_dir, 'points'))

    def overlay(state):
        path = case_file(case_dir, state)
        if not os.path.exists(path):
            return None
        return load_mesh(path, view, step_size=step_size, prefiltered=True, top_only=True)

    return build_html(verts, faces, rest, frames, title, frames_interp=frames_interp,
                      expi_mesh=overlay('expi'),
                      normal_mesh=overlay('normal'), sigma=sigma, reg=reg,
                      default_method=default_method, biharm_weight=biharm_weight)
