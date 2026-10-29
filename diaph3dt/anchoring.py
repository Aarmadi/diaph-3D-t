"""Anchoring of the 2D+t trajectories on the inspiratory template mesh.

Each tracked point is mapped to 3D (scanner frame) at every frame. Its 3D trajectory
crosses the upper (lung side) surface of the inspiratory mesh; the crossing gives the
point's position on the template (barycentric coordinates on the crossed face). When the
trajectory crosses the surface several times, the actual crossing closest to the mean of
the crossings is kept."""

import json

import numpy as np

from .geometry import model_pts_to_ras
from .template import mesh_signed_volume


def _segment_triangle_intersections(p0, p1, verts, faces, eps=1e-9):
    """Vectorised Moller-Trumbore: intersections of the segment [p0, p1] with all faces.
    Returns (points (K, 3), faces (K, 3)) of the K crossed faces."""
    v0 = verts[faces[:, 0]]
    edge1 = verts[faces[:, 1]] - v0
    edge2 = verts[faces[:, 2]] - v0
    d = p1 - p0
    h = np.cross(d[None, :], edge2)
    a = np.einsum('ij,ij->i', edge1, h)
    valid = np.abs(a) > eps
    f = np.divide(1.0, a, out=np.zeros_like(a), where=valid)
    s = p0[None, :] - v0
    u = f * np.einsum('ij,ij->i', s, h)
    valid &= (u >= -eps) & (u <= 1 + eps)
    q = np.cross(s, edge1)
    v = f * (q @ d)
    valid &= (v >= -eps) & (u + v <= 1 + eps)
    t = f * np.einsum('ij,ij->i', edge2, q)
    valid &= (t >= -eps) & (t <= 1 + eps)   # segment, not the infinite line
    idx = np.where(valid)[0]
    if len(idx) == 0:
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)
    return p0[None, :] + t[idx][:, None] * d[None, :], faces[idx]


def _barycentric_weights(p, v0, v1, v2):
    """Barycentric weights (w0, w1, w2) of p in the triangle (v0, v1, v2)."""
    e0, e1, e2 = v1 - v0, v2 - v0, p - v0
    d00, d01, d11 = np.dot(e0, e0), np.dot(e0, e1), np.dot(e1, e1)
    d20, d21 = np.dot(e2, e0), np.dot(e2, e1)
    denom = d00 * d11 - d01 * d01
    w1 = (d11 * d20 - d01 * d21) / denom
    w2 = (d00 * d21 - d01 * d20) / denom
    return 1.0 - w1 - w2, w1, w2


def trajectory_crossings(traj_ras, mesh_verts, faces):
    """All crossings of a polyline (T, 3) with the given faces. Returns (points, faces)."""
    pts, fcs = [], []
    for a, b in zip(traj_ras[:-1], traj_ras[1:]):
        p, f = _segment_triangle_intersections(np.asarray(a), np.asarray(b), mesh_verts, faces)
        if len(p):
            pts.append(p)
            fcs.append(f)
    if not pts:
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)
    return np.concatenate(pts), np.concatenate(fcs)


def anchor_trajectory(traj_ras, mesh_verts, faces):
    """Anchor of a trajectory on the template: the crossing closest to the mean of all
    crossings (an observed position, never the mean itself).

    Returns None without crossing, else dict(position, face, weights, all_crossings)."""
    pts, fcs = trajectory_crossings(traj_ras, mesh_verts, faces)
    if len(pts) == 0:
        return None
    k = int(np.argmin(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))
    face = fcs[k]
    weights = np.array(_barycentric_weights(pts[k], *mesh_verts[face]))
    return dict(position=pts[k], face=face, weights=weights, all_crossings=pts)


def anchor_plane(track, mesh_verts, anchor_faces, min_confidence=0.8, t_offset=0.0):
    """Anchor the tracked points of one plane on the template.

    For each point not flagged as an island, only the frames where the CoTracker
    confidence is >= min_confidence are kept (a frame is dropped, not the whole point).
    The time of each observation is shifted by t_offset (acquisition delay of the plane).

    Returns (control_points, diagnostics): control_points is a list of
    dict(rest_pos, by_t={t: RAS position}); diagnostics feeds build_anchor_html."""
    tracks_rc, conf, island = track['tracks_rc'], track['confidence'], track['island_mask']
    control_points, trajectories = [], []
    n_low_conf = n_no_cross = 0
    for i in range(track['n_points']):
        if island[i]:
            continue
        frames = [t for t in range(track['nt']) if conf[t, i] >= min_confidence]
        if len(frames) < 2:
            n_low_conf += 1
            continue
        traj = model_pts_to_ras(tracks_rc[frames, i], track['aff_seg'], track['ni'], track['nj'])
        anchor = anchor_trajectory(traj, mesh_verts, anchor_faces)
        if anchor is None:
            n_no_cross += 1
            continue
        control_points.append(dict(rest_pos=anchor['position'],
                                   by_t={t + t_offset: p for t, p in zip(frames, traj)}))
        trajectories.append(dict(
            pts3d=[[round(float(c), 2) for c in p] for p in traj],
            crossings=[[round(float(c), 2) for c in p] for p in anchor['all_crossings']],
            anchor=[round(float(c), 2) for c in anchor['position']]))
    n_init = int(track['n_points'] - island.sum())
    print(f'  [plane {track["plane"]}] {n_init} point(s) after island removal, '
          f'{n_low_conf} with < 2 confident frames, {n_no_cross} never crossing the '
          f'template -> {len(control_points)} anchored')
    return control_points, dict(pi=track['plane'], nInit=n_init, trajectories=trajectories)


CSS_ANCHOR = """
body{margin:0;font-family:-apple-system,Segoe UI,sans-serif;background:#0d1117;color:#cdd9e5}
h1{text-align:center;padding:16px 0 4px;font-size:1.3rem;margin:0}
.sub{text-align:center;color:#888;font-size:.85rem;margin-bottom:10px;max-width:820px;
  margin-left:auto;margin-right:auto}
#wrap{display:flex;flex-direction:column;align-items:center;padding:0 16px 24px}
canvas{background:#0d1117;border:1px solid #30363d;border-radius:6px;cursor:grab}
canvas:active{cursor:grabbing}
.ctrl{display:flex;gap:14px;align-items:center;flex-wrap:wrap;
  margin-top:10px;max-width:900px;justify-content:center}
.ctrl label{font-size:.82rem;color:#ccc;display:flex;align-items:center;gap:6px}
.ctrl select{background:#21262d;color:#ccc;border:1px solid #444;border-radius:4px;
  padding:4px 8px;max-width:600px}
.ctrl input[type=checkbox]{accent-color:#58a6ff;width:15px;height:15px}
.ctrl button{padding:5px 14px;background:#21262d;border:1px solid #444;border-radius:4px;
  color:#ccc;cursor:pointer}
.ctrl button:hover{background:#2d333b}
.info{font-family:monospace;font-size:.8rem;color:#888;margin-top:8px;text-align:center}
.leg{font-size:.75rem;color:#666;margin-top:10px;max-width:900px;text-align:center}
.leg b{color:#aaa}
"""


JS_ANCHOR = r"""
var VERTS=__VERTS__, FACES=__FACES__, FLIP_NORMALS=__FLIP_NORMALS__;
var PLANES=__PLANES__;
var curPlane=0;
var showSurface=false, showWire=true, showAxes=false, showCrossings=true;

function trajColor(idx, n){
  var h=(idx/Math.max(n,1))%1, s=0.7, v=0.95;
  var i=Math.floor(h*6), f=h*6-i, p=v*(1-s), q=v*(1-f*s), t=v*(1-(1-f)*s);
  var rgb=[[v,t,p],[q,v,p],[p,v,t],[p,q,v],[t,p,v],[v,p,q]][i%6];
  return 'rgb('+Math.round(rgb[0]*255)+','+Math.round(rgb[1]*255)+','+Math.round(rgb[2]*255)+')';
}

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

var bboxMin=[Infinity,Infinity,Infinity], bboxMax=[-Infinity,-Infinity,-Infinity];
for(var i=0;i<VERTS.length;i++){
  for(var k=0;k<3;k++){
    if(VERTS[i][k]<bboxMin[k])bboxMin[k]=VERTS[i][k];
    if(VERTS[i][k]>bboxMax[k])bboxMax[k]=VERTS[i][k];
  }
}
var CENTER=[(bboxMin[0]+bboxMax[0])/2,(bboxMin[1]+bboxMax[1])/2,(bboxMin[2]+bboxMax[2])/2];
var DIAG=Math.hypot(bboxMax[0]-bboxMin[0],bboxMax[1]-bboxMin[1],bboxMax[2]-bboxMin[2])||1;

var rotY=0.6, rotX=-0.35, rotZ=0.0, zoom=1.0, dragging=false, lastX=0, lastY=0;

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
      render();
    },30);
  }else if(!(autoRot.y||autoRot.x||autoRot.z)&&autoRotTimer){
    clearInterval(autoRotTimer);autoRotTimer=null;
  }
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

function drawMeshSurface(meshVerts, meshFaces, flipNormals, baseR, baseG, baseB, alpha){
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
    ctx.fillStyle="rgba("+Math.round(baseR*shade)+","+Math.round(baseG*shade)+","+Math.round(baseB*shade)+","+alpha+")";
    var pa=proj[face[0]], pb=proj[face[1]], pc=proj[face[2]];
    ctx.beginPath();
    ctx.moveTo(pa[0],pa[1]); ctx.lineTo(pb[0],pb[1]); ctx.lineTo(pc[0],pc[1]);
    ctx.closePath(); ctx.fill();
  }
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
  var grad=ctx.createRadialGradient(WPX/2,HPX/2,0,WPX/2,HPX/2,Math.max(WPX,HPX)*0.75);
  grad.addColorStop(0,"#242c3a"); grad.addColorStop(1,"#0c1018");
  ctx.fillStyle=grad; ctx.fillRect(0,0,WPX,HPX);

  if(showSurface){
    drawMeshSurface(VERTS, FACES, FLIP_NORMALS, SURF_R, SURF_G, SURF_B, 0.55);
  }
  var proj=VERTS.map(project);
  if(showWire){
    ctx.strokeStyle=showSurface?"rgba(5,8,12,0.35)":"rgba(88,166,255,0.5)";
    ctx.lineWidth=1;
    ctx.beginPath();
    for(var e=0;e<EDGES.length;e++){
      var ed=EDGES[e], a=proj[ed[0]], b=proj[ed[1]];
      ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]);
    }
    ctx.stroke();
  }

  var plane=PLANES[curPlane];
  var nTraj=plane.trajectories.length, nMulti=0;
  plane.trajectories.forEach(function(traj,idx){
    var col=trajColor(idx, nTraj);
    if(traj.crossings.length>1)nMulti++;

    ctx.strokeStyle=col; ctx.lineWidth=1.4; ctx.globalAlpha=0.85;
    ctx.beginPath();
    traj.pts3d.forEach(function(p,i){
      var pr=project(p);
      if(i===0)ctx.moveTo(pr[0],pr[1]); else ctx.lineTo(pr[0],pr[1]);
    });
    ctx.stroke();
    ctx.globalAlpha=1.0;

    if(showCrossings){
      traj.crossings.forEach(function(c){
        var cpr=project(c);
        ctx.beginPath(); ctx.arc(cpr[0],cpr[1],2.6,0,6.283);
        ctx.strokeStyle=col; ctx.lineWidth=1.2; ctx.stroke();
      });
    }

    var apr=project(traj.anchor);
    ctx.beginPath(); ctx.arc(apr[0],apr[1],4.5,0,6.283);
    ctx.fillStyle=col; ctx.fill();
    ctx.strokeStyle="#fff"; ctx.lineWidth=1.2; ctx.stroke();
  });

  if(showAxes)drawAxes();

  var infoEl=document.getElementById("info");
  if(infoEl){
    infoEl.textContent="plane "+plane.pi+"   "+plane.nInit+" point(s) (islands excluded) -- "+
      nTraj+" anchored ("+(plane.nInit?Math.round(100*nTraj/plane.nInit):0)+"%) -- "+
      nMulti+" with several crossings";
  }
}

function selectPlane(v){
  curPlane=parseInt(v,10);
  render();
}
function toggleLayer(name,val){
  if(name==="surface")showSurface=val;
  else if(name==="wire")showWire=val;
  else if(name==="axes")showAxes=val;
  else if(name==="crossings")showCrossings=val;
  render();
}

var canvas, ctx, WPX, HPX;
function initViewer(){
  canvas=document.getElementById("cv");
  ctx=canvas.getContext("2d");
  WPX=canvas.width; HPX=canvas.height;
  canvas.addEventListener("mousedown",function(e){dragging=true;lastX=e.clientX;lastY=e.clientY;});
  window.addEventListener("mouseup",function(){dragging=false;});
  window.addEventListener("mousemove",function(e){
    if(dragging){
      rotY+=(e.clientX-lastX)*0.01;
      rotX+=(e.clientY-lastY)*0.01;
      rotX=Math.max(-1.5,Math.min(1.5,rotX));
      lastX=e.clientX; lastY=e.clientY;
      render();
    }
  });
  canvas.addEventListener("wheel",function(e){
    e.preventDefault();
    zoom*=(e.deltaY<0)?1.08:0.93;
    zoom=Math.max(0.2,Math.min(6,zoom));
    render();
  },{passive:false});

  render();
}
"""


def build_anchor_html(subject, view, mesh_verts, mesh_faces, plane_anchors):
    """Interactive viewer of the anchoring: 3D trajectories, their crossings with the
    template and the retained anchor. plane_anchors: diagnostics of anchor_plane."""
    flip_normals = mesh_signed_volume(mesh_verts, mesh_faces) < 0
    plane_opts = ''.join(
        f'<option value="{i}">plane {pl["pi"]}  ({len(pl["trajectories"])}/{pl["nInit"]} '
        'anchored)</option>'
        for i, pl in enumerate(plane_anchors))
    js = (JS_ANCHOR
          .replace('__VERTS__', json.dumps([[round(float(x), 3) for x in v] for v in mesh_verts]))
          .replace('__FACES__', json.dumps(mesh_faces.tolist()))
          .replace('__FLIP_NORMALS__', 'true' if flip_normals else 'false')
          .replace('__PLANES__', json.dumps(plane_anchors, separators=(',', ':'))))
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>3D anchoring - {subject}</title>'
        f'<style>{CSS_ANCHOR}</style><script>{js}</script></head>'
        '<body onload="initViewer()"><div id="wrap">'
        f'<h1>Trajectory / template anchoring - {subject} {view}</h1>'
        '<p class="sub">3D trajectories of the tracked points and their anchors on the '
        'upper surface of the inspiratory template (drag = rotate, wheel = zoom).</p>'
        '<div class="ctrl">'
        f'<label>Plane: <select onchange="selectPlane(this.value)">{plane_opts}</select></label>'
        '</div>'
        '<canvas id="cv" width="700" height="700"></canvas>'
        '<div class="ctrl">'
        '<label><input type="checkbox" onchange="toggleLayer(\'surface\',this.checked)"> '
        'Shaded surface</label>'
        '<label><input type="checkbox" checked onchange="toggleLayer(\'wire\',this.checked)"> '
        'Wireframe</label>'
        '<label><input type="checkbox" onchange="toggleLayer(\'axes\',this.checked)"> '
        'RAS axes</label>'
        '<label><input type="checkbox" checked onchange="toggleLayer(\'crossings\',this.checked)"> '
        'All crossings</label>'
        '</div>'
        '<div class="ctrl">'
        '<span style="color:#888;font-size:.8rem">Auto-rotation:</span>'
        '<button id="rot-play-y" onclick="toggleAutoRot(\'y\')">&#9654; Horizontal</button>'
        '<button id="rot-play-x" onclick="toggleAutoRot(\'x\')">&#9654; Vertical</button>'
        '<button id="rot-play-z" onclick="toggleAutoRot(\'z\')">&#9654; Roll</button>'
        '</div>'
        '<p class="info" id="info"></p>'
        '<p class="leg">One colour per point. <b>Hollow circles</b>: every crossing of the '
        'trajectory with the upper surface (a point followed over several breathing cycles '
        'can cross it several times). <b>Filled point</b>: retained anchor, the actual '
        'crossing closest to the mean of the crossings.</p>'
        '</div></body></html>'
    )
