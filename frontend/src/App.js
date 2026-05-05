import React, { useState, useEffect, useCallback, useRef } from 'react';
import Papa from 'papaparse';
import axios from 'axios';

let Plotly = null;
async function getPlotly() {
  if (!Plotly) Plotly = (await import('plotly.js-dist-min')).default;
  return Plotly;
}

const API = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const ORS_KEY = 'eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjM1ZDhjN2E0ZDNjNzQ3NWE5ZWU0YTViZDY0NmY2NTAxIiwiaCI6Im11cm11cjY0In0=';
const OSM = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
const OSM_ATTR = '&copy; <a href="https://openstreetmap.org">OpenStreetMap</a>';

const MAPS = {};

function waitForLeaflet() {
  return new Promise(resolve => {
    if (window.L) { resolve(window.L); return; }
    const check = setInterval(() => {
      if (window.L) { clearInterval(check); resolve(window.L); }
    }, 50);
    setTimeout(() => { clearInterval(check); resolve(null); }, 10000);
  });
}

async function mkMap(id, lat=28.6139, lon=77.2090, zoom=13) {
  const L = await waitForLeaflet();
  if (!L) { console.error('Leaflet not available'); return null; }
  if (MAPS[id]) { try { MAPS[id].remove(); } catch(_){} delete MAPS[id]; }
  let el = document.getElementById(id);
  if (!el) {
    await new Promise(r => setTimeout(r, 100));
    el = document.getElementById(id);
  }
  if (!el) { console.error('Map div not found:', id); return null; }
  delete el._leaflet_id;
  el.innerHTML = '';
  const map = L.map(el, { center:[lat,lon], zoom });
  L.tileLayer(OSM, { attribution:OSM_ATTR, maxZoom:19 }).addTo(map);
  const lg = L.layerGroup().addTo(map);
  MAPS[id] = map;
  return { map, lg };
}
function rmMap(id) {
  if (MAPS[id]) { try { MAPS[id].remove(); } catch(_){} delete MAPS[id]; }
}
function dotIcon(color) {
  return window.L.divIcon({
    html:`<div style="width:14px;height:14px;border-radius:50%;background:${color};border:3px solid #fff;box-shadow:0 2px 8px rgba(0,0,0,.4)"></div>`,
    iconSize:[14,14],iconAnchor:[7,7],className:'',
  });
}
function lblIcon(color, txt) {
  return window.L.divIcon({
    html:`<div style="background:${color};color:#fff;padding:3px 10px;border-radius:20px;font-size:11px;font-family:'JetBrains Mono',monospace;white-space:nowrap;box-shadow:0 2px 6px rgba(0,0,0,.3)">${txt}</div>`,
    iconAnchor:[35,12],className:'',
  });
}

function haversine(a,b,c,d){
  const R=6371000,r=v=>v*Math.PI/180;
  return 2*R*Math.asin(Math.sqrt(Math.min(1,Math.max(0,
    Math.sin(r(c-a)/2)**2+Math.cos(r(a))*Math.cos(r(c))*Math.sin(r(d-b)/2)**2))));
}

function makeDemoData(n=120){
  const rows=[],start=new Date('2024-06-15T09:00:00');
  for(let i=0;i<n;i++){
    const t=(i/(n-1))*4*Math.PI,rn=()=>Math.random()+Math.random()-1;
    const lb=28.6139+0.003*(t/(4*Math.PI)),lo=77.2090+0.004*Math.sin(t/2);
    const e=Math.random()<0.14,eb=e?0.0012:0;
    rows.push({
      timestamp:new Date(start.getTime()+i*6000).toISOString().replace('T',' ').slice(0,19),
      latitude:+(lb+rn()*0.00035+rn()*eb).toFixed(7),
      longitude:+(lo+rn()*0.00035+rn()*eb).toFixed(7),
      corrected_latitude:+(lb+rn()*0.00006).toFixed(7),
      corrected_longitude:+(lo+rn()*0.00006).toFixed(7),
      speed:+Math.max(5,Math.min(80,25+18*Math.sin(t)+rn()*2)).toFixed(2),
      error_label:e?1:0,
    });
  }
  return rows;
}

function computeMetrics(rows){
  const n=rows.length,nE=rows.filter(r=>+r.error_label===1).length;
  const rl=rows.reduce((s,r)=>s+ +r.corrected_latitude,0)/n;
  const ro=rows.reduce((s,r)=>s+ +r.corrected_longitude,0)/n;
  const rd=rows.reduce((s,r)=>s+haversine(+r.latitude,+r.longitude,rl,ro),0)/n;
  const cd=rows.reduce((s,r)=>s+haversine(+r.corrected_latitude,+r.corrected_longitude,rl,ro),0)/n;
  return{
    total:n,nErrors:nE,errorPct:+(nE/n*100).toFixed(1),
    avgSpeed:+(rows.reduce((s,r)=>s+ +r.speed,0)/n).toFixed(1),
    improvement:+Math.min(99.9,Math.max(0,(rd-cd)/rd*100)).toFixed(1),
    avgDisp:+(rows.reduce((s,r)=>s+haversine(+r.latitude,+r.longitude,+r.corrected_latitude,+r.corrected_longitude),0)/n).toFixed(2),
  };
}

async function fetchORS(slon,slat,elon,elat){
  const r=await fetch('https://api.openrouteservice.org/v2/directions/driving-car/geojson',{
    method:'POST',
    headers:{'Authorization':ORS_KEY,'Content-Type':'application/json'},
    body:JSON.stringify({coordinates:[[slon,slat],[elon,elat]],instructions:false}),
  });
  if(!r.ok) throw new Error(`ORS ${r.status}: ${r.statusText}`);
  const d=await r.json();
  if(!d.features||!d.features[0]) throw new Error('No route returned by ORS');
  return d.features[0].geometry.coordinates.map(c=>({lat:c[1],lon:c[0]}));
}

async function chart(id,traces,layout){
  const P=await getPlotly();
  if(!document.getElementById(id)) return;
  await P.react(id,traces,{
    paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(250,250,252,1)',
    font:{family:'JetBrains Mono,monospace',size:11},
    margin:{l:55,r:20,t:30,b:50},...layout
  },{responsive:true});
}

const CSS=`
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500&family=Syne:wght@600;700;800&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
:root{--bg:#f4f6fb;--sb:#0d1117;--card:#fff;--brd:#e2e8f0;--tx:#1a202c;--mt:#64748b;--ff:'Syne',sans-serif;--fm:'JetBrains Mono',monospace;}
body{background:var(--bg);color:var(--tx);font-family:var(--ff);}
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-thumb{background:#cbd5e1;border-radius:99px;}
.layout{display:flex;min-height:100vh;}
.sb{width:260px;min-width:260px;background:var(--sb);color:#e2e8f0;display:flex;flex-direction:column;position:sticky;top:0;height:100vh;overflow-y:auto;}
.main{flex:1;padding:28px 32px;overflow-x:hidden;}
.sbh{padding:24px 20px 16px;border-bottom:1px solid rgba(255,255,255,.07);}
.logo{font-size:18px;font-weight:700;color:#fff;}
.logo-sub{font-family:var(--fm);font-size:10px;color:rgba(255,255,255,.35);margin-top:4px;letter-spacing:.1em;}
.sbs{padding:16px 20px;border-bottom:1px solid rgba(255,255,255,.06);}
.sbl{font-family:var(--fm);font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:rgba(255,255,255,.4);margin-bottom:10px;}
.sbf{padding:16px 20px;margin-top:auto;font-family:var(--fm);font-size:9px;color:rgba(255,255,255,.22);line-height:1.6;}
.navs{display:flex;flex-direction:column;gap:4px;padding:14px;}
.nt{padding:9px 14px;border-radius:8px;border:none;background:transparent;color:rgba(255,255,255,.5);font-family:var(--fm);font-size:11px;cursor:pointer;text-align:left;transition:all .18s;display:flex;align-items:center;gap:8px;}
.nt:hover{background:rgba(255,255,255,.06);color:rgba(255,255,255,.85);}
.nt.on{background:rgba(41,128,185,.22);color:#5dade2;border-left:3px solid #2980b9;}
.upld{border:1.5px dashed rgba(255,255,255,.18);border-radius:10px;padding:16px 12px;text-align:center;cursor:pointer;transition:all .2s;}
.upld:hover,.upld.drag{border-color:rgba(41,128,185,.7);background:rgba(41,128,185,.07);}
.tglr{display:flex;align-items:center;gap:8px;margin-bottom:8px;cursor:pointer;}
.tglr input{width:15px;height:15px;accent-color:#2980b9;cursor:pointer;}
.tgl-lbl{font-family:var(--fm);font-size:11px;color:rgba(255,255,255,.65);}
.dot{width:10px;height:10px;border-radius:50%;flex-shrink:0;}
input[type=range]{width:100%;accent-color:#2980b9;}
.btn{width:100%;margin-top:6px;padding:10px;border-radius:8px;border:none;cursor:pointer;background:linear-gradient(135deg,#2980b9,#1a5276);color:#fff;font-family:var(--fm);font-size:12px;font-weight:500;transition:opacity .2s,transform .15s;display:flex;align-items:center;justify-content:center;gap:6px;}
.btn:hover{opacity:.88;transform:translateY(-1px);}
.btn:disabled{opacity:.4;cursor:not-allowed;transform:none;}
.btn-g{background:linear-gradient(135deg,#27ae60,#1e8449);}
.btn-r{background:linear-gradient(135deg,#c0392b,#922b21);}
.btn-sm{width:auto;padding:9px 20px;margin-top:0;}
.spin{display:inline-block;width:12px;height:12px;border:2px solid rgba(255,255,255,.3);border-top-color:#fff;border-radius:50%;animation:sp .7s linear infinite;}
@keyframes sp{to{transform:rotate(360deg);}}
.ph{margin-bottom:24px;}
.pt{font-size:26px;font-weight:800;color:var(--tx);margin-bottom:4px;}
.ps{font-family:var(--fm);font-size:12px;color:var(--mt);}
.bnr{margin-bottom:16px;padding:10px 14px;border-radius:10px;font-family:var(--fm);font-size:11px;display:flex;align-items:center;gap:8px;}
.bi{background:#eff6ff;border:1px solid #bfdbfe;color:#1d4ed8;}
.bo{background:#f0fdf4;border:1px solid #bbf7d0;color:#15803d;}
.bw{background:#fffbeb;border:1px solid #fcd34d;color:#92400e;}
.be{background:#fef2f2;border:1px solid #fecaca;color:#991b1b;}
.mcs{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:24px;}
.mc4{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px;}
.mc{background:var(--card);border:1px solid var(--brd);border-radius:14px;padding:14px 16px;}
.mcl{font-family:var(--fm);font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:var(--mt);margin-bottom:6px;}
.mcv{font-size:1.6rem;font-weight:700;line-height:1;margin-bottom:4px;}
.mcd{font-family:var(--fm);font-size:10px;color:var(--mt);}
.cb{color:#2980b9;}.cr{color:#e74c3c;}.cg{color:#27ae60;}.cw{color:#f39c12;}.ctx{color:var(--tx);}
.sec{margin-bottom:24px;}
.stt{font-size:14px;font-weight:700;color:var(--tx);margin-bottom:4px;}
.stc{font-family:var(--fm);font-size:11px;color:var(--mt);margin-bottom:12px;}
.dvd{border:none;border-top:1px solid var(--brd);margin:20px 0;}
.crd{background:var(--card);border:1px solid var(--brd);border-radius:14px;padding:16px 18px;}
.twoc{display:grid;grid-template-columns:3fr 2fr;gap:16px;}
.etrow{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px;}
.bdg{padding:5px 12px;border-radius:20px;font-family:var(--fm);font-size:10px;font-weight:500;}
.bmp{background:#fee2e2;color:#b91c1c;}
.bio{background:#fef3c7;color:#b45309;}
.bge{background:#ede9fe;color:#6d28d9;}
.bok{background:#dcfce7;color:#15803d;}
.ldot{width:10px;height:10px;border-radius:50%;background:#e74c3c;}
.ldot.on{background:#27ae60;animation:pulse 1.2s infinite;}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:.4;}}
.tw{overflow-x:auto;border-radius:10px;border:1px solid var(--brd);}
table{width:100%;border-collapse:collapse;font-family:var(--fm);font-size:11px;}
thead th{background:#f8fafc;padding:10px 12px;text-align:left;font-weight:500;color:var(--mt);border-bottom:1px solid var(--brd);white-space:nowrap;}
tbody tr{border-bottom:1px solid #f1f5f9;}
tbody tr:hover{background:#f8fafc;}
tbody tr.er{background:#fffbeb;}
td{padding:8px 12px;color:var(--tx);white-space:nowrap;}
.eb{background:#fef3c7;color:#b45309;padding:2px 8px;border-radius:20px;font-size:10px;}
.gb{background:#dcfce7;color:#15803d;padding:2px 8px;border-radius:20px;font-size:10px;}
.bdl{padding:8px 18px;border-radius:8px;border:1px solid var(--brd);background:#fff;color:var(--tx);font-family:var(--fm);font-size:11px;cursor:pointer;margin-top:10px;}
.bdl:hover{background:#f1f5f9;}
.prog{height:3px;border-radius:99px;background:rgba(255,255,255,.08);margin-top:8px;overflow:hidden;}
.pf{height:100%;border-radius:99px;background:#2980b9;transition:width .3s;}
.mleg{padding:10px 14px;border-bottom:1px solid var(--brd);font-family:var(--fm);font-size:11px;color:var(--mt);display:flex;gap:14px;flex-wrap:wrap;align-items:center;}
.li{display:flex;align-items:center;gap:5px;}
@media(max-width:900px){
  .layout{flex-direction:column;}.sb{width:100%;height:auto;position:static;}
  .mcs,.mc4{grid-template-columns:repeat(2,1fr);}
  .twoc{grid-template-columns:1fr;}.main{padding:16px;}
}
`;

// ── Dashboard Map ─────────────────────────────────────────────
function DashMap({rows,showRaw,showCorr,showErr}){
  const mRef=useRef(null),lgRef=useRef(null);
  useEffect(()=>{
    if(!rows.length) return;
    async function draw(){
      if(!mRef.current){
        const cla=rows.reduce((s,r)=>s+ +r.corrected_latitude,0)/rows.length;
        const clo=rows.reduce((s,r)=>s+ +r.corrected_longitude,0)/rows.length;
        const r=await mkMap('d-map',cla,clo,13);
        if(!r) return;
        mRef.current=r.map; lgRef.current=r.lg;
      }
      const L=window.L,lg=lgRef.current,map=mRef.current;
      if(!L||!lg||!map) return;
      lg.clearLayers();
      if(showRaw) L.polyline(rows.map(r=>[+r.latitude,+r.longitude]),{color:'#e74c3c',weight:2.5,opacity:.85}).addTo(lg);
      if(showCorr) L.polyline(rows.map(r=>[+r.corrected_latitude,+r.corrected_longitude]),{color:'#2980b9',weight:3.5,opacity:.95}).addTo(lg);
      if(showErr) rows.filter(r=>+r.error_label===1).forEach(r=>{
        L.circleMarker([+r.latitude,+r.longitude],{radius:7,color:'#f39c12',fillColor:'#f39c12',fillOpacity:.85,weight:2}).bindPopup('⚠ GNSS Error').addTo(lg);
      });
      L.marker([+rows[0].corrected_latitude,+rows[0].corrected_longitude],{icon:lblIcon('#27ae60','▶ Start')}).addTo(lg);
      L.marker([+rows[rows.length-1].corrected_latitude,+rows[rows.length-1].corrected_longitude],{icon:lblIcon('#8e44ad','■ End')}).addTo(lg);
      map.fitBounds(rows.map(r=>[+r.corrected_latitude,+r.corrected_longitude]),{padding:[30,30]});
    }
    draw();
  },[rows,showRaw,showCorr,showErr]);
  useEffect(()=>()=>rmMap('d-map'),[]);
  return <div id="d-map" style={{height:480,width:'100%'}}/>;
}

// ── Route Engine ──────────────────────────────────────────────
function RouteEngine({apiBase, showTrue, showRaw, showCorr, showBad}){
  const [phase,setPhase]=useState('idle');
  const [spt,setSpt]=useState(null);
  const [ept,setEpt]=useState(null);
  const [res,setRes]=useState(null);
  const [err,setErr]=useState('');
  const mRef=useRef(null),lgRef=useRef(null),phRef=useRef('idle');
  useEffect(()=>{phRef.current=phase;},[phase]);

  useEffect(()=>{
    async function init(){
      const r=await mkMap('r-map',28.6139,77.2090,12);
      if(!r) return;
      mRef.current=r.map; lgRef.current=r.lg;
      r.map.on('click',e=>{
        const pt={lat:e.latlng.lat,lon:e.latlng.lng};
        if(phRef.current==='picking_end'){setEpt(pt);setPhase('ready');}
        else{setSpt(pt);setEpt(null);setRes(null);setPhase('picking_end');}
      });
    }
    init();
    return()=>rmMap('r-map');
  },[]);

  useEffect(()=>{
    const L=window.L,lg=lgRef.current,map=mRef.current;
    if(!L||!lg||!map||res) return;
    lg.clearLayers();
    if(spt) L.marker([spt.lat,spt.lon],{icon:lblIcon('#27ae60','▶ Start')}).addTo(lg);
    if(ept) L.marker([ept.lat,ept.lon],{icon:lblIcon('#c0392b','■ End')}).addTo(lg);
    if(spt&&ept){
      L.polyline([[spt.lat,spt.lon],[ept.lat,ept.lon]],{color:'#2980b9',weight:2,dashArray:'6,5',opacity:.5}).addTo(lg);
      map.fitBounds([[spt.lat,spt.lon],[ept.lat,ept.lon]],{padding:[60,60]});
    }
  },[spt,ept,res]);

  // ── Redraw map when layer toggles change ──────────────────
  useEffect(()=>{
    const L=window.L,lg=lgRef.current,map=mRef.current;
    if(!L||!lg||!map||!res) return;
    lg.clearLayers();
    const T=res.trajectory;
    if(showTrue) L.polyline(T.map(p=>[p.true_lat,p.true_lon]),{color:'#8e44ad',weight:5,opacity:.75}).addTo(lg);
    if(showRaw)  L.polyline(T.map(p=>[p.raw_lat,p.raw_lon]),{color:'#e74c3c',weight:2.5,dashArray:'5,4',opacity:.9}).addTo(lg);
    if(showCorr) L.polyline(T.map(p=>[p.corr_lat,p.corr_lon]),{color:'#27ae60',weight:4,opacity:.95}).addTo(lg);
    if(showBad)  T.filter(p=>p.flagged).forEach(p=>{
      L.circleMarker([p.raw_lat,p.raw_lon],{radius:7,color:'#f39c12',fillColor:'#f39c12',fillOpacity:.85,weight:2})
        .bindPopup(`⚠ Prob: ${p.error_proba.toFixed(3)}<br>Err: ${p.raw_err_m.toFixed(1)} m`).addTo(lg);
    });
    L.marker([T[0].true_lat,T[0].true_lon],{icon:lblIcon('#27ae60','▶ Start')}).addTo(lg);
    L.marker([T[T.length-1].true_lat,T[T.length-1].true_lon],{icon:lblIcon('#c0392b','■ End')}).addTo(lg);
    map.fitBounds(T.map(p=>[p.true_lat,p.true_lon]),{padding:[30,30]});
  },[res, showTrue, showRaw, showCorr, showBad]);

  useEffect(()=>{
    if(!res) return;
    setTimeout(()=>chart('re-chart',[
      {x:res.error_series.map(e=>e.i),y:res.error_series.map(e=>e.raw),mode:'lines',name:'Raw Error (m)',line:{color:'#e74c3c',width:2},fill:'tozeroy',fillcolor:'rgba(231,76,60,0.1)'},
      {x:res.error_series.map(e=>e.i),y:res.error_series.map(e=>e.corrected),mode:'lines',name:'Corrected Error (m)',line:{color:'#27ae60',width:2.5},fill:'tozeroy',fillcolor:'rgba(39,174,96,0.12)'},
    ],{xaxis:{title:'GPS Epoch'},yaxis:{title:'Error (m)'},height:260,hovermode:'x unified',legend:{orientation:'h',y:1.1}}),300);
  },[res]);

  async function run(){
    if(!spt||!ept) return;
    setPhase('loading');setErr('');setRes(null);
    try{
      const wp=await fetchORS(spt.lon,spt.lat,ept.lon,ept.lat);
      if(!wp||wp.length<4) throw new Error('ORS returned too few points. Try points on a main road.');
      const r=await axios.post(`${apiBase}/api/route/correct`,{waypoints:wp});
      setRes(r.data);setPhase('done');
    }catch(e){setErr(e?.response?.data?.detail||e?.message||'Error');setPhase('error');}
  }
  function reset(){setPhase('idle');setSpt(null);setEpt(null);setRes(null);setErr('');const lg=lgRef.current;if(lg)lg.clearLayers();}

  const instr={idle:'👆 Click "Pick Points" then click the map to set START',picking_end:'✅ Start set! Click the map to set END',ready:'✅ Both set — click Run AI Correction',loading:'⏳ Fetching road + running AI...',done:'✅ Done! Results below',error:'❌ Error below'};

  return(<>
    <div className="ph"><div className="pt">🗺️ Route Correction Engine</div><div className="ps">Click 2 points → ORS fetches real road → AI corrects GNSS errors</div></div>
    <div className={`bnr ${phase==='done'?'bo':phase==='error'?'be':'bi'}`} style={{marginBottom:12}}>{instr[phase]||instr.idle}</div>
    <div style={{display:'flex',gap:10,marginBottom:14,flexWrap:'wrap',alignItems:'center'}}>
      <button className="btn btn-sm" onClick={()=>{setPhase('picking_end');setSpt(null);setEpt(null);setRes(null);setErr('');const lg=lgRef.current;if(lg)lg.clearLayers();}}>📍 Pick Points</button>
      <button className="btn btn-g btn-sm" disabled={!spt||!ept||phase==='loading'} onClick={run}>
        {phase==='loading'?<><span className="spin"/> Running...</>:'🚀 Run AI Correction'}
      </button>
      <button className="bdl" style={{marginTop:0}} onClick={reset}>🗑️ Reset</button>
      {spt&&<span style={{fontFamily:'var(--fm)',fontSize:11,color:'var(--mt)'}}>
        <span style={{color:'#27ae60'}}>▶</span> {spt.lat.toFixed(5)},{spt.lon.toFixed(5)}
        {ept&&<> &nbsp;<span style={{color:'#c0392b'}}>■</span> {ept.lat.toFixed(5)},{ept.lon.toFixed(5)}</>}
      </span>}
    </div>
    <div className="crd" style={{padding:0,overflow:'hidden',marginBottom:20}}>
      <div className="mleg">
        🗺️ Click anywhere on map
        {res&&(<>
          <span className="li"><span style={{width:16,height:4,background:'#8e44ad',display:'inline-block',borderRadius:2}}/> True road</span>
          <span className="li"><span style={{width:16,height:4,background:'#e74c3c',display:'inline-block',borderRadius:2}}/> Raw GPS</span>
          <span className="li"><span style={{width:16,height:4,background:'#27ae60',display:'inline-block',borderRadius:2}}/> AI Corrected</span>
          <span className="li"><span style={{width:10,height:10,background:'#f39c12',display:'inline-block',borderRadius:'50%'}}/> Bad points</span>
        </>)}
      </div>
      <div id="r-map" style={{height:500,width:'100%'}}/>
    </div>
    {err&&<div className="bnr be" style={{marginBottom:16}}>⚠️ {err}</div>}
    {res&&(<>
      {/* Before / After comparison cards */}
<div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:12,marginBottom:16}}>
  
  {/* BEFORE card */}
  <div style={{background:'#fff5f5',border:'1px solid #fecaca',borderRadius:14,padding:'14px 16px'}}>
    <div style={{fontFamily:'var(--fm)',fontSize:10,letterSpacing:'.12em',textTransform:'uppercase',color:'#ef4444',marginBottom:10,fontWeight:600}}>
      📡 Before AI Correction
    </div>
    <div style={{display:'flex',flexDirection:'column',gap:8}}>
      <div>
        <div style={{fontFamily:'var(--fm)',fontSize:10,color:'#94a3b8',marginBottom:2}}>Distance</div>
        <div style={{fontSize:'1.4rem',fontWeight:700,color:'#e74c3c'}}>{res.summary.raw_distance_km} km</div>
      </div>
      <div>
        <div style={{fontFamily:'var(--fm)',fontSize:10,color:'#94a3b8',marginBottom:2}}>Est. Travel Time</div>
        <div style={{fontSize:'1.4rem',fontWeight:700,color:'#e74c3c'}}>{res.summary.raw_time_min} min</div>
      </div>
      <div>
        <div style={{fontFamily:'var(--fm)',fontSize:10,color:'#94a3b8',marginBottom:2}}>Avg GPS Error</div>
        <div style={{fontSize:'1.4rem',fontWeight:700,color:'#e74c3c'}}>{res.summary.avg_raw_error_m} m</div>
      </div>
    </div>
  </div>

  {/* AFTER card */}
  <div style={{background:'#f0fdf4',border:'1px solid #bbf7d0',borderRadius:14,padding:'14px 16px'}}>
    <div style={{fontFamily:'var(--fm)',fontSize:10,letterSpacing:'.12em',textTransform:'uppercase',color:'#22c55e',marginBottom:10,fontWeight:600}}>
      🤖 After AI Correction
    </div>
    <div style={{display:'flex',flexDirection:'column',gap:8}}>
      <div>
        <div style={{fontFamily:'var(--fm)',fontSize:10,color:'#94a3b8',marginBottom:2}}>Distance</div>
        <div style={{fontSize:'1.4rem',fontWeight:700,color:'#27ae60'}}>{res.summary.corr_distance_km} km</div>
      </div>
      <div>
        <div style={{fontFamily:'var(--fm)',fontSize:10,color:'#94a3b8',marginBottom:2}}>Est. Travel Time</div>
        <div style={{fontSize:'1.4rem',fontWeight:700,color:'#27ae60'}}>{res.summary.corr_time_min} min</div>
      </div>
      <div>
        <div style={{fontFamily:'var(--fm)',fontSize:10,color:'#94a3b8',marginBottom:2}}>Avg GPS Error</div>
        <div style={{fontSize:'1.4rem',fontWeight:700,color:'#27ae60'}}>{res.summary.avg_corr_error_m} m</div>
      </div>
    </div>
  </div>
</div>

{/* Summary stats row */}
<div className="mc4" style={{marginBottom:16}}>
  {[
    ['Road Points', res.summary.total_points, 'ctx', 'from ORS'],
    ['Bad GPS', res.summary.flagged_points, 'cw', `${((res.summary.flagged_points/res.summary.total_points)*100).toFixed(1)}%`],
    ['Improvement', `${res.summary.improvement_pct}%`, 'cg', 'error reduced'],
    ['Model', res.summary.model_used, 'cb', 'active'],
  ].map(([l,v,c,d])=>(
    <div key={l} className="mc"><div className="mcl">{l}</div><div className={`mcv ${c}`} style={{fontSize:'1.1rem'}}>{v}</div><div className="mcd">{d}</div></div>
  ))}
</div>
      <div className="etrow">
        <span className="bdg bmp">🔴 Multipath: {res.error_types.multipath_points}</span>
        <span className="bdg bio">🟡 Iono: {res.error_types.iono_affected_points}</span>
        <span className="bdg bge">🟣 Geometry: {res.error_types.poor_geometry_points}</span>
        <span className="bdg bok">✅ {res.summary.model_used}</span>
      </div>
      <div className="crd" style={{marginBottom:20}}>
        <div style={{fontFamily:'var(--fm)',fontSize:12,fontWeight:600,marginBottom:8}}>Error: Raw vs AI-Corrected (metres)</div>
        <div id="re-chart" style={{width:'100%'}}/>
      </div>
      <div className="sec"><div className="stt">📋 Route Data</div>
        <div className="tw"><table>
          <thead><tr>{['#','True Lat','True Lon','Raw Lat','Raw Lon','Corr Lat','Corr Lon','Prob','Raw(m)','Corr(m)','Status'].map(h=><th key={h}>{h}</th>)}</tr></thead>
          <tbody>{res.trajectory.slice(0,60).map((p,i)=>(
            <tr key={i} className={p.flagged?'er':''}>
              <td>{p.i}</td><td>{p.true_lat.toFixed(6)}</td><td>{p.true_lon.toFixed(6)}</td>
              <td>{p.raw_lat.toFixed(6)}</td><td>{p.raw_lon.toFixed(6)}</td>
              <td>{p.corr_lat.toFixed(6)}</td><td>{p.corr_lon.toFixed(6)}</td>
              <td>{p.error_proba.toFixed(3)}</td><td>{p.raw_err_m.toFixed(2)}</td><td>{p.corr_err_m.toFixed(2)}</td>
              <td>{p.flagged?<span className="eb">⚠ Bad</span>:<span className="gb">✓ Good</span>}</td>
            </tr>
          ))}</tbody>
        </table></div>
      </div>
    </>)}
  </>);
}

// ── Live GPS ──────────────────────────────────────────────────
function LiveGPS({apiBase}){
  const [status,setStatus]=useState('idle');
  const [pts,setPts]=useState([]);
  const [err,setErr]=useState('');
  const [lc,setLc]=useState(null);
  const [wid,setWid]=useState(null);
  const mRef=useRef(null),lgRef=useRef(null);
  useEffect(()=>{
    async function init(){
      const r=await mkMap('l-map',28.6139,77.2090,14);
      if(!r) return;
      mRef.current=r.map; lgRef.current=r.lg;
    }
    init();
    return()=>rmMap('l-map');
  },[]);
  useEffect(()=>{
    const L=window.L,lg=lgRef.current,map=mRef.current;
    if(!L||!lg||!map||!pts.length) return;
    lg.clearLayers();
    L.polyline(pts.map(p=>[p.lat,p.lon]),{color:'#e74c3c',weight:2.5}).addTo(lg);
    const cr=pts.filter(p=>p.clat);
    if(cr.length) L.polyline(cr.map(p=>[p.clat,p.clon]),{color:'#27ae60',weight:3}).addTo(lg);
    const last=pts[pts.length-1];
    L.circleMarker([last.lat,last.lon],{radius:10,color:'#f39c12',fillColor:'#f39c12',fillOpacity:1,weight:2}).bindPopup('📍 Here').addTo(lg);
    map.setView([last.lat,last.lon]);
  },[pts]);
  function start(){
    if(!navigator.geolocation){setErr('Geolocation not supported');setStatus('error');return;}
    setStatus('tracking');setErr('');setPts([]);
    const id=navigator.geolocation.watchPosition(async pos=>{
      const{latitude:lat,longitude:lon,accuracy,speed}=pos.coords;
      const pt={lat,lon,accuracy,speed:speed||0,ts:Date.now()};
      try{
        const r=await axios.post(`${apiBase}/api/predict`,{raw_lat:lat,raw_lon:lon,mean_cn0:30,min_elevation:35,multipath_sum:1,iono_delay:1,tropo_delay:0.5,num_satellites:8,displacement:5,speed_derived:speed||2,acceleration:0.1,heading_change:5,pos_variance:1});
        pt.clat=r.data.corrected_lat;pt.clon=r.data.corrected_lon;pt.ep=r.data.error_probability;pt.bad=r.data.is_bad_gps;pt.q=r.data.quality_score;pt.imp=r.data.improvement_pct;
        setLc(r.data);
      }catch(_){}
      setPts(prev=>[...prev.slice(-199),pt]);
    },e=>{setErr(e.message);setStatus('error');},{enableHighAccuracy:true,maximumAge:0,timeout:10000});
    setWid(id);
  }
  function stop(){if(wid!==null)navigator.geolocation.clearWatch(wid);setStatus('idle');setWid(null);}
  return(<>
    <div className="ph"><div className="pt">📍 Live GPS Correction</div><div className="ps">Real-time AI correction using your device GPS</div></div>
    <div className="bnr bw" style={{marginBottom:16}}>⚠️ Open on your phone for best results. Each fix is sent to AI backend for correction.</div>
    <div className="crd" style={{marginBottom:20}}>
      <div style={{display:'flex',alignItems:'center',gap:8,fontFamily:'var(--fm)',fontSize:12,marginBottom:12}}>
        <div className={`ldot${status==='tracking'?' on':''}`}/>
        {status==='idle'?'Not tracking':status==='tracking'?`Tracking — ${pts.length} points`:'Error'}
      </div>
      {err&&<div className="bnr be" style={{marginBottom:8}}>⚠️ {err}</div>}
      <div style={{display:'flex',gap:10}}>
        <button className="btn btn-g" style={{flex:1}} disabled={status==='tracking'} onClick={start}>
          {status==='tracking'?<><span className="spin"/> Tracking...</>:'▶ Start'}
        </button>
        <button className="btn btn-r" style={{flex:1}} disabled={status!=='tracking'} onClick={stop}>⏹ Stop</button>
      </div>
    </div>
    {lc&&<div className="mc4" style={{marginBottom:20}}>
      {[['Quality',`${lc.quality_score}%`,lc.quality_score>70?'cg':'cw',''],['Error Prob',lc.error_probability.toFixed(3),lc.is_bad_gps?'cr':'cg',lc.is_bad_gps?'Bad GPS':'Good'],['Raw Err',`${lc.raw_error_m} m`,'cr','est'],['Corr Err',`${lc.corrected_error_m} m`,'cg',`${lc.improvement_pct}% better`]].map(([l,v,c,d])=>(
        <div key={l} className="mc"><div className="mcl">{l}</div><div className={`mcv ${c}`}>{v}</div><div className="mcd">{d}</div></div>
      ))}
    </div>}
    {pts.length>0&&(()=>{const last=pts[pts.length-1];return(
      <div style={{fontFamily:'var(--fm)',fontSize:11,color:'var(--mt)',background:'#f8fafc',padding:'8px 12px',borderRadius:8,marginBottom:10}}>
        📍 {last.lat.toFixed(6)}, {last.lon.toFixed(6)} · Accuracy: {last.accuracy?`±${last.accuracy.toFixed(1)} m`:'N/A'}
        {last.clat&&<> · ✅ {last.clat.toFixed(6)}, {last.clon.toFixed(6)}</>}
      </div>
    );})()}
    <div className="crd" style={{padding:0,overflow:'hidden'}}>
      <div className="mleg">
        <span className="li"><span style={{width:16,height:4,background:'#e74c3c',display:'inline-block'}}/> Raw GPS</span>
        <span className="li"><span style={{width:16,height:4,background:'#27ae60',display:'inline-block'}}/> AI Corrected</span>
        <span className="li"><span style={{width:10,height:10,background:'#f39c12',display:'inline-block',borderRadius:'50%'}}/> Current position</span>
      </div>
      <div id="l-map" style={{height:450,width:'100%'}}/>
    </div>
  </>);
}

// ── App ───────────────────────────────────────────────────────
export default function App(){
  const [tab,setTab]=useState('dashboard');
  const [rows,setRows]=useState([]);
  const [src,setSrc]=useState('');
  const [file,setFile]=useState(null);
  const [drag,setDrag]=useState(false);
  const [loading,setLoading]=useState(false);
  const [prog,setProg]=useState(0);

  // Dashboard map layer toggles
  const [sR,setSR]=useState(true);
  const [sC,setSC]=useState(true);
  const [sE,setSE]=useState(true);

  // Route Engine map layer toggles — NEW
  const [rShowTrue,setRShowTrue]=useState(true);
  const [rShowRaw,setRShowRaw]=useState(true);
  const [rShowCorr,setRShowCorr]=useState(true);
  const [rShowBad,setRShowBad]=useState(true);

  const [sMin,setSMin]=useState(0);
  const [sMax,setSMax]=useState(120);
  const [onlyE,setOnlyE]=useState(false);
  const [maxR,setMaxR]=useState(50);
  const dRef=useRef();

  const filt=rows.filter(r=>+r.speed>=sMin&&+r.speed<=sMax);
  const met=filt.length?computeMetrics(filt):null;
  const tRows=(onlyE?filt.filter(r=>+r.error_label===1):filt).slice(0,maxR);

  useEffect(()=>{setRows(makeDemoData(120));setSrc('demo');},[]);

  useEffect(()=>{
    if(tab!=='dashboard'||!filt.length) return;
    const t=setTimeout(async()=>{
      const nE=filt.filter(r=>+r.error_label===1).length;
      await chart('c-speed',[
        {x:filt.map(r=>r.timestamp),y:filt.map(r=>+r.speed),mode:'lines',line:{color:'#2980b9',width:2},name:'Speed'},
        {x:filt.filter(r=>+r.error_label===1).map(r=>r.timestamp),y:filt.filter(r=>+r.error_label===1).map(r=>+r.speed),mode:'markers',marker:{color:'#f39c12',size:10,symbol:'x'},name:'⚠ Error'},
      ],{xaxis:{title:'Time',tickfont:{size:10}},yaxis:{title:'Speed (km/h)'},height:300,hovermode:'x unified',legend:{orientation:'h',y:1.1}});
      await chart('c-dist',[{type:'bar',x:['Normal','Error'],y:[filt.length-nE,nE],marker:{color:['#2980b9','#f39c12']},text:[filt.length-nE,nE],textposition:'auto'}],
        {xaxis:{title:'Type'},yaxis:{title:'Count'},height:280,showlegend:false});
      const disp=filt.map(r=>haversine(+r.latitude,+r.longitude,+r.corrected_latitude,+r.corrected_longitude));
      await chart('c-disp',[{type:'bar',x:filt.map(r=>r.timestamp),y:disp.map(v=>+v.toFixed(3)),marker:{color:filt.map(r=>+r.error_label===1?'#f39c12':'#2980b9')}}],
        {xaxis:{title:'Time',tickfont:{size:10}},yaxis:{title:'Displacement (m)'},height:260,showlegend:false});
    },200);
    return()=>clearTimeout(t);
  },[tab,filt,sR,sC,sE,sMin,sMax]);

  const parseFile=useCallback(f=>{
    if(!f?.name.endsWith('.csv')) return;
    setFile(f);setLoading(true);setProg(20);
    Papa.parse(f,{header:true,skipEmptyLines:true,complete:res=>{
      setProg(80);
      const needed=['timestamp','latitude','longitude','corrected_latitude','corrected_longitude','speed','error_label'];
      const filled=res.data.map((r,i)=>{const out={...r};if(!out.timestamp)out.timestamp=new Date(Date.now()+i*6000).toISOString();needed.forEach(k=>{if(out[k]===undefined)out[k]=0;});return out;});
      setRows(filled);setSrc('file');setProg(100);setLoading(false);
    },error:()=>setLoading(false)});
  },[]);

  return(<>
    <style>{CSS}</style>
    <div className="layout">
      <aside className="sb">
        <div className="sbh"><div className="logo">🛰️ GNSS AI</div><div className="logo-sub">AI CORRECTION DASHBOARD</div></div>
        <div className="navs">
          <div className="sbl" style={{paddingLeft:0}}>Navigation</div>
          {[['dashboard','📊','Dashboard'],['route','🗺️','Route Engine'],['live','📍','Live GPS']].map(([id,ic,lb])=>(
            <button key={id} className={`nt${tab===id?' on':''}`} onClick={()=>setTab(id)}>{ic} {lb}</button>
          ))}
        </div>

        {/* ── Dashboard sidebar ── */}
        {tab==='dashboard'&&(<>
          <div className="sbs">
            <div className="sbl">📂 Data</div>
            <div className={`upld${drag?' drag':''}`} onDrop={e=>{e.preventDefault();setDrag(false);parseFile(e.dataTransfer.files[0]);}} onDragOver={e=>{e.preventDefault();setDrag(true);}} onDragLeave={()=>setDrag(false)} onClick={()=>dRef.current.click()}>
              <input ref={dRef} type="file" accept=".csv" style={{display:'none'}} onChange={e=>parseFile(e.target.files[0])}/>
              <div style={{fontSize:22,marginBottom:6}}>📁</div>
              <div style={{fontFamily:'var(--fm)',fontSize:11,color:'rgba(255,255,255,.5)'}}>{file?file.name:'Upload CSV'}</div>
              <div style={{fontFamily:'var(--fm)',fontSize:9,color:'rgba(255,255,255,.22)',marginTop:3}}>{file?`${(file.size/1024).toFixed(1)} KB`:'Drop or click'}</div>
            </div>
            {loading&&<div className="prog"><div className="pf" style={{width:`${prog}%`}}/></div>}
            <button className="btn" style={{marginTop:8}} onClick={()=>{setRows(makeDemoData(120));setSrc('demo');setFile(null);}}>🧪 Demo Data</button>
          </div>
          <div className="sbs">
            <div className="sbl">🗺️ Map Layers</div>
            {[[sR,setSR,'#e74c3c','Raw GPS'],[sC,setSC,'#2980b9','Corrected GPS'],[sE,setSE,'#f39c12','Error Points']].map(([v,s,c,l])=>(
              <label key={l} className="tglr"><input type="checkbox" checked={v} onChange={e=>s(e.target.checked)}/><span className="dot" style={{background:c}}/><span className="tgl-lbl">{l}</span></label>
            ))}
          </div>
          <div className="sbs">
            <div className="sbl">⚡ Speed Filter</div>
            <div style={{fontFamily:'var(--fm)',fontSize:10,color:'rgba(255,255,255,.5)',marginBottom:6}}>{sMin}–{sMax} km/h</div>
            <input type="range" min={0} max={120} step={5} value={sMin} onChange={e=>setSMin(+e.target.value)} style={{marginBottom:6}}/>
            <input type="range" min={0} max={120} step={5} value={sMax} onChange={e=>setSMax(+e.target.value)}/>
          </div>
        </>)}

        {/* ── Route Engine sidebar — NEW map layer toggles ── */}
        {tab==='route'&&<>
          <div className="sbs">
            <div className="sbl">ℹ️ Steps</div>
            <div style={{fontFamily:'var(--fm)',fontSize:10,color:'rgba(255,255,255,.45)',lineHeight:1.9}}>
              1. Click Pick Points<br/>
              2. Click map → START<br/>
              3. Click map → END<br/>
              4. Run AI Correction<br/>
              5. See corrected route
            </div>
          </div>
          <div className="sbs">
            <div className="sbl">🗺️ Map Layers</div>
            {[
              [rShowTrue, setRShowTrue, '#8e44ad', 'True Road'],
              [rShowRaw,  setRShowRaw,  '#e74c3c', 'Raw GPS'],
              [rShowCorr, setRShowCorr, '#27ae60', 'AI Corrected'],
              [rShowBad,  setRShowBad,  '#f39c12', 'Bad Points'],
            ].map(([v,s,c,l])=>(
              <label key={l} className="tglr">
                <input type="checkbox" checked={v} onChange={e=>s(e.target.checked)}/>
                <span className="dot" style={{background:c}}/>
                <span className="tgl-lbl">{l}</span>
              </label>
            ))}
          </div>
        </>}

        {tab==='live'&&<div className="sbs"><div className="sbl">ℹ️ Info</div><div style={{fontFamily:'var(--fm)',fontSize:10,color:'rgba(255,255,255,.45)',lineHeight:1.9}}>Uses device GPS<br/>Sends to AI backend<br/>Shows corrected path<br/>Best on mobile</div></div>}

        <div className="sbf">AI-Based GNSS Correction<br/>B.Tech Second Year · GLA University 2025-26<br/>React · Leaflet · Plotly · FastAPI · XGBoost</div>
      </aside>

      <main className="main">
        {tab==='dashboard'&&(<>
          <div className="ph"><div className="pt">📊 GNSS AI Correction Dashboard</div><div className="ps">Raw GPS · AI-corrected trajectories · GNSS error analysis</div></div>
          {src==='demo'&&<div className="bnr bi">🧪 Demo data — New Delhi route. Upload a CSV to use your own data.</div>}
          {src==='file'&&<div className="bnr bo">📂 Loaded <strong>{file?.name}</strong> — {rows.length.toLocaleString()} rows</div>}
          {met&&<div className="mcs">
            {[['Total',met.total.toLocaleString(),'ctx',''],['Errors',met.nErrors,'cw',`${met.errorPct}%`],['Avg Speed',`${met.avgSpeed} km/h`,'cb',''],['Improvement',`${met.improvement}%`,'cg','vs raw'],['Avg Disp',`${met.avgDisp} m`,'cb','raw→corr']].map(([l,v,c,d])=>(
              <div key={l} className="mc"><div className="mcl">{l}</div><div className={`mcv ${c}`}>{v}</div>{d&&<div className="mcd">{d}</div>}</div>
            ))}
          </div>}
          <hr className="dvd"/>
          <div className="sec">
            <div className="stt">🗺️ GPS Trajectory Map</div>
            <div className="stc"><span style={{color:'#e74c3c'}}>●</span> Red = Raw GPS &nbsp;·&nbsp; <span style={{color:'#2980b9'}}>●</span> Blue = AI-corrected &nbsp;·&nbsp; <span style={{color:'#f39c12'}}>●</span> Orange = Errors</div>
            <div className="crd" style={{padding:0,overflow:'hidden'}}>
              {filt.length>0&&<DashMap rows={filt} showRaw={sR} showCorr={sC} showErr={sE}/>}
            </div>
          </div>
          <hr className="dvd"/>
          <div className="sec"><div className="stt">📈 Analytics</div>
            <div className="twoc">
              <div className="crd"><div style={{fontFamily:'var(--fm)',fontSize:12,fontWeight:600,marginBottom:8}}>Speed vs Time</div><div id="c-speed" style={{width:'100%'}}/></div>
              <div className="crd"><div style={{fontFamily:'var(--fm)',fontSize:12,fontWeight:600,marginBottom:8}}>Error Distribution</div><div id="c-dist" style={{width:'100%'}}/></div>
            </div>
          </div>
          <div className="sec"><div className="crd"><div style={{fontFamily:'var(--fm)',fontSize:12,fontWeight:600,marginBottom:8}}>Point-wise Displacement</div><div id="c-disp" style={{width:'100%'}}/></div></div>
          <hr className="dvd"/>
          <div className="sec"><div className="stt">📋 Data Preview</div>
            <div style={{display:'flex',gap:16,marginBottom:10,flexWrap:'wrap',alignItems:'center'}}>
              <label style={{display:'flex',alignItems:'center',gap:6,fontFamily:'var(--fm)',fontSize:11,cursor:'pointer'}}>
                <input type="checkbox" checked={onlyE} onChange={e=>setOnlyE(e.target.checked)}/> Errors only
              </label>
              <div style={{display:'flex',alignItems:'center',gap:6,fontFamily:'var(--fm)',fontSize:11}}>
                Rows: <select style={{fontFamily:'var(--fm)',fontSize:11,padding:'4px 8px',border:'1px solid var(--brd)',borderRadius:6}} value={maxR} onChange={e=>setMaxR(+e.target.value)}>
                  {[25,50,100,200].map(v=><option key={v}>{v}</option>)}
                </select>
              </div>
            </div>
            <div className="tw"><table>
              <thead><tr>{['Timestamp','Lat','Lon','Corr Lat','Corr Lon','Speed','Status'].map(h=><th key={h}>{h}</th>)}</tr></thead>
              <tbody>{tRows.map((r,i)=>(
                <tr key={i} className={+r.error_label===1?'er':''}>
                  <td>{String(r.timestamp).slice(0,19)}</td>
                  <td>{Number(r.latitude).toFixed(6)}</td><td>{Number(r.longitude).toFixed(6)}</td>
                  <td>{Number(r.corrected_latitude).toFixed(6)}</td><td>{Number(r.corrected_longitude).toFixed(6)}</td>
                  <td>{Number(r.speed).toFixed(2)}</td>
                  <td>{+r.error_label===1?<span className="eb">⚠ Error</span>:<span className="gb">✓ Normal</span>}</td>
                </tr>
              ))}</tbody>
            </table></div>
            <button className="bdl" onClick={()=>{const h=Object.keys(filt[0]||{}).join(','),b=filt.map(r=>Object.values(r).join(',')).join('\n'),a=document.createElement('a');a.href=URL.createObjectURL(new Blob([h+'\n'+b],{type:'text/csv'}));a.download='gnss.csv';a.click();}}>⬇️ Download CSV</button>
          </div>
        </>)}

        {tab==='route'&&<RouteEngine
          apiBase={API}
          showTrue={rShowTrue}
          showRaw={rShowRaw}
          showCorr={rShowCorr}
          showBad={rShowBad}
        />}

        {tab==='live'&&<LiveGPS apiBase={API}/>}
        <hr className="dvd"/>
        <div style={{fontFamily:'var(--fm)',fontSize:10,color:'var(--mt)',textAlign:'center',paddingBottom:24}}>AI-Based GNSS Correction · B.Tech Second Year · GLA University 2025-26</div>
      </main>
    </div>
  </>);
}