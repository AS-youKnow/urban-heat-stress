"""
module7_dashboard.py
─────────────────────────────────────────────────────────────────────────────
MODULE 7 : INTERACTIVE STREAMLIT WEB DASHBOARD
─────────────────────────────────────────────────────────────────────────────

A full-featured, professional-grade Streamlit web application that ties all
six upstream modules together in one interactive interface.

UI Features:
  ┌────────────────────────────────────────────────────────────────┐
  │  SIDEBAR                                                       │
  │  • Data source selector (GEE live / Synthetic demo)            │
  │  • Budget slider (N cells to optimise)                         │
  │  • Optimization weight sliders (α severity, β sensitivity)     │
  │  • Scenario checkboxes                                         │
  │  • Run Pipeline button                                         │
  ├────────────────────────────────────────────────────────────────┤
  │  TAB 1 — HEAT MAP                                              │
  │  • Folium choropleth of LST / Gi* hotspot categories           │
  │  • Target cells overlaid as red markers                        │
  ├────────────────────────────────────────────────────────────────┤
  │  TAB 2 — MODEL PERFORMANCE                                     │
  │  • MAE / RMSE / R² metric cards                                │
  │  • Actual vs Predicted scatter plot                            │
  │  • Residual distribution histogram                             │
  ├────────────────────────────────────────────────────────────────┤
  │  TAB 3 — SHAP ANALYSIS                                         │
  │  • Beeswarm summary plot                                       │
  │  • Feature importance bar chart                                │
  │  • Interpretive narrative                                      │
  ├────────────────────────────────────────────────────────────────┤
  │  TAB 4 — SCENARIO COMPARISON                                   │
  │  • Mean / Median / P95 ΔT bar charts                           │
  │  • % Cells cooled comparison                                   │
  │  • Summary metrics table                                       │
  ├────────────────────────────────────────────────────────────────┤
  │  TAB 5 — OPTIMIZATION RESULTS                                  │
  │  • Strategy allocation donut chart                             │
  │  • Top-N target cells table                                    │
  │  • Priority score map overlay                                  │
  └────────────────────────────────────────────────────────────────┘

Run:
    streamlit run module7_dashboard.py

Dependencies: streamlit, folium, streamlit-folium, plotly, pandas, numpy,
              matplotlib, shap, xgboost, scikit-learn
─────────────────────────────────────────────────────────────────────────────
"""

import os
import io
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import HeatMap, MarkerCluster

# Local modules
from config import CFG, get_synthetic_dataframe
from module1_data_ingestion import run_ingestion
from module2_hotspot_clustering import run_hotspot_analysis, heat_vulnerability_profile
from module3_predictive_modeling import run_modeling
from module4_shap_interpretation import run_shap_analysis
from module5_scenario_simulation import run_simulation
from module6_optimization import run_full_optimization

matplotlib.use("Agg")
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ─── 3D LANDING PAGE ─────────────────────────────────────────────────────────

def render_landing_page():
    """Full-screen 3D animated landing page with rotating Earth globe."""
    st.components.v1.html("""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Urban Heat Stress AI</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    background: #020810;
    font-family: 'Segoe UI', sans-serif;
    overflow: hidden;
    width: 100vw; height: 100vh;
  }
  #canvas-container {
    position: fixed; top:0; left:0;
    width:100%; height:100%;
    z-index: 0;
  }
  .overlay {
    position: fixed; top:0; left:0;
    width:100%; height:100%;
    z-index: 10;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    pointer-events: none;
  }
  .badge {
    background: rgba(0,212,170,0.12);
    border: 1px solid rgba(0,212,170,0.4);
    border-radius: 50px;
    padding: 6px 18px;
    color: #00d4aa;
    font-size: 12px;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 22px;
    animation: fadeInDown 1s ease forwards;
    opacity: 0;
  }
  .title {
    font-size: clamp(28px, 5vw, 56px);
    font-weight: 800;
    text-align: center;
    line-height: 1.15;
    margin-bottom: 16px;
    animation: fadeInDown 1s 0.2s ease forwards;
    opacity: 0;
  }
  .title .line1 { color: #ffffff; display:block; }
  .title .line2 {
    display: block;
    background: linear-gradient(90deg, #ff4b4b, #ff8c00, #00d4aa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .subtitle {
    color: #8b949e;
    font-size: clamp(13px, 2vw, 17px);
    text-align: center;
    max-width: 560px;
    line-height: 1.7;
    margin-bottom: 40px;
    animation: fadeInDown 1s 0.4s ease forwards;
    opacity: 0;
    padding: 0 20px;
  }
  .stats-row {
    display: flex;
    gap: 28px;
    margin-bottom: 44px;
    animation: fadeInDown 1s 0.6s ease forwards;
    opacity: 0;
  }
  .stat {
    text-align: center;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 14px 22px;
    backdrop-filter: blur(10px);
  }
  .stat .num {
    font-size: 24px;
    font-weight: 700;
    color: #00d4aa;
  }
  .stat .label {
    font-size: 11px;
    color: #8b949e;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 4px;
  }
  .btn {
    pointer-events: all;
    cursor: pointer;
    padding: 16px 48px;
    font-size: 16px;
    font-weight: 700;
    color: #020810;
    background: linear-gradient(135deg, #00d4aa, #00a88a);
    border: none;
    border-radius: 50px;
    letter-spacing: 1px;
    text-transform: uppercase;
    box-shadow: 0 0 40px rgba(0,212,170,0.4),
                0 0 80px rgba(0,212,170,0.15);
    transition: all 0.3s ease;
    animation: fadeInUp 1s 0.8s ease forwards, pulse 2.5s 2s infinite;
    opacity: 0;
    position: relative;
    overflow: hidden;
  }
  .btn::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: linear-gradient(45deg,
      transparent 40%, rgba(255,255,255,0.3) 50%, transparent 60%);
    transform: translateX(-100%);
    transition: transform 0.6s ease;
  }
  .btn:hover::before { transform: translateX(100%); }
  .btn:hover {
    transform: scale(1.06);
    box-shadow: 0 0 60px rgba(0,212,170,0.6),
                0 0 120px rgba(0,212,170,0.25);
  }
  .btn:active { transform: scale(0.97); }
  .orb {
    position: fixed;
    border-radius: 50%;
    filter: blur(80px);
    z-index: 1;
    animation: orbFloat 8s ease-in-out infinite;
  }
  .orb1 {
    width:400px; height:400px;
    background: radial-gradient(circle, rgba(255,75,75,0.18) 0%, transparent 70%);
    top:-100px; right:-100px;
    animation-delay: 0s;
  }
  .orb2 {
    width:350px; height:350px;
    background: radial-gradient(circle, rgba(0,212,170,0.15) 0%, transparent 70%);
    bottom:-80px; left:-80px;
    animation-delay: -4s;
  }
  .orb3 {
    width:280px; height:280px;
    background: radial-gradient(circle, rgba(74,158,255,0.12) 0%, transparent 70%);
    top:40%; left:10%;
    animation-delay: -2s;
  }
  .hotspot-ring {
    position: fixed;
    border-radius: 50%;
    border: 2px solid rgba(255,75,75,0.6);
    animation: ring-pulse 2.5s ease-out infinite;
    z-index: 5;
  }
  @keyframes ring-pulse {
    0%   { transform: scale(0.5); opacity:1; }
    100% { transform: scale(2.5); opacity:0; }
  }
  @keyframes fadeInDown {
    from { opacity:0; transform:translateY(-20px); }
    to   { opacity:1; transform:translateY(0); }
  }
  @keyframes fadeInUp {
    from { opacity:0; transform:translateY(20px); }
    to   { opacity:1; transform:translateY(0); }
  }
  @keyframes pulse {
    0%,100% { box-shadow: 0 0 40px rgba(0,212,170,0.4), 0 0 80px rgba(0,212,170,0.15); }
    50%      { box-shadow: 0 0 60px rgba(0,212,170,0.7), 0 0 120px rgba(0,212,170,0.3); }
  }
  @keyframes orbFloat {
    0%,100% { transform: translate(0,0) scale(1); }
    33%      { transform: translate(30px,-20px) scale(1.05); }
    66%      { transform: translate(-20px,15px) scale(0.95); }
  }
  .particles-canvas {
    position:fixed; top:0; left:0;
    width:100%; height:100%;
    z-index:2; pointer-events:none;
  }
  .corner-info {
    position:fixed; bottom:24px; left:50%;
    transform:translateX(-50%);
    color:rgba(139,148,158,0.6);
    font-size:12px;
    letter-spacing:2px;
    text-transform:uppercase;
    z-index:20;
    animation: fadeInDown 1s 1.2s ease forwards;
    opacity:0;
  }
  .isro-badge {
    position:fixed; top:20px; right:24px;
    background:rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.1);
    border-radius:8px;
    padding:8px 16px;
    color:#8b949e;
    font-size:12px;
    z-index:20;
    letter-spacing:1px;
  }
  .scanline {
    position:fixed; top:0; left:0;
    width:100%; height:2px;
    background: linear-gradient(90deg, transparent, #00d4aa, transparent);
    animation: scan 4s linear infinite;
    z-index:20;
    opacity:0.4;
  }
  @keyframes scan {
    from { top: -2px; }
    to   { top: 100%; }
  }
</style>
</head>
<body>

<!-- Ambient orbs -->
<div class="orb orb1"></div>
<div class="orb orb2"></div>
<div class="orb orb3"></div>

<!-- Scan line effect -->
<div class="scanline"></div>

<!-- ISRO badge -->
<div class="isro-badge">ISRO Research Project 2025-26</div>

<!-- 3D Globe Canvas -->
<div id="canvas-container">
  <canvas id="globe"></canvas>
</div>

<!-- Particle canvas -->
<canvas class="particles-canvas" id="particles"></canvas>

<!-- Hotspot pulse rings -->
<div class="hotspot-ring" style="width:24px;height:24px;top:38%;left:62%;animation-delay:0s;"></div>
<div class="hotspot-ring" style="width:20px;height:20px;top:45%;left:58%;animation-delay:0.8s;"></div>
<div class="hotspot-ring" style="width:18px;height:18px;top:35%;left:65%;animation-delay:1.6s;"></div>

<!-- Main overlay content -->
<div class="overlay">
  <div class="badge">AI / ML Powered</div>
  <h1 class="title">
    <span class="line1">Urban Heat Stress</span>
    <span class="line2">Hotspot Prediction System</span>
  </h1>
  <p class="subtitle">
    Satellite-driven AI platform for predicting, clustering, simulating
    and optimizing urban heat stress hotspots across the globe using
    Landsat 8, XGBoost &amp; SHAP analysis.
  </p>
  <div class="stats-row">
    <div class="stat"><div class="num">50+</div><div class="label">World Cities</div></div>
    <div class="stat"><div class="num">6</div><div class="label">AI Modules</div></div>
    <div class="stat"><div class="num">100m</div><div class="label">Grid Scale</div></div>
    <div class="stat"><div class="num">3</div><div class="label">Scenarios</div></div>
  </div>
  <button class="btn" onclick="launch()">&#9658; Launch Dashboard</button>
</div>

<div class="corner-info">Scroll or click to explore &nbsp; &bull; &nbsp; Powered by Google Earth Engine</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
// ── 3D Globe ──────────────────────────────────────────────────────────────────
const canvas = document.getElementById('globe');
const renderer = new THREE.WebGLRenderer({ canvas, antialias:true, alpha:true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);

const scene  = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.1, 1000);
camera.position.set(0, 0, 2.8);

// Stars
const starGeo = new THREE.BufferGeometry();
const starCount = 6000;
const starPos = new Float32Array(starCount * 3);
for(let i=0;i<starCount*3;i++) starPos[i] = (Math.random()-0.5)*200;
starGeo.setAttribute('position', new THREE.BufferAttribute(starPos,3));
const starMat = new THREE.PointsMaterial({color:0xffffff, size:0.12, transparent:true, opacity:0.7});
scene.add(new THREE.Points(starGeo, starMat));

// Earth sphere
const earthGeo = new THREE.SphereGeometry(1, 64, 64);
const earthMat = new THREE.MeshPhongMaterial({
  color: 0x1a4a7a,
  emissive: 0x0a1a3a,
  specular: 0x4a9eff,
  shininess: 25,
  wireframe: false,
});
const earth = new THREE.Mesh(earthGeo, earthMat);
scene.add(earth);

// Continent dots (latitude/longitude grid)
const continentPoints = [
  // India region
  [20,77],[22,79],[24,81],[26,83],[28,77],[18,74],[16,74],[23,87],[13,80],[12,77],
  // East Asia
  [35,135],[36,140],[37,127],[31,121],[23,113],[39,116],[30,120],[32,118],
  // Europe
  [51,-0.1],[48,2],[41,-4],[52,13],[41,12],[55,37],[40,22],[38,23],
  // Africa
  [30,31],[6,3],[-1,37],[-26,28],[15,32],[-4,15],[9,38],[5,-0.3],
  // Americas
  [40,-74],[34,-118],[41,-87],[30,-95],[19,-99],[-23,-46],[-22,-43],[-34,-58],[4,-74],[-12,-77],
  // Australia
  [-33,151],[-37,144],[-27,153]
];

const dotGeo  = new THREE.SphereGeometry(0.012, 6, 6);
const dotMat  = new THREE.MeshBasicMaterial({ color: 0x2a6aaa });
continentPoints.forEach(([lat,lon]) => {
  const phi   = (90 - lat) * Math.PI/180;
  const theta = (lon + 180) * Math.PI/180;
  const dot   = new THREE.Mesh(dotGeo, dotMat);
  dot.position.set(
    -Math.sin(phi)*Math.cos(theta),
     Math.cos(phi),
     Math.sin(phi)*Math.sin(theta)
  );
  earth.add(dot);
});

// Heat hotspot markers (red glowing dots)
const hotspots = [
  [28.6,77.2,'Delhi'],[19.1,72.9,'Mumbai'],[22.6,88.4,'Kolkata'],
  [30,31.2,'Cairo'],[6.5,3.4,'Lagos'],[24.7,46.7,'Riyadh'],
  [25.2,55.3,'Dubai'],[31.2,121.5,'Shanghai'],[35.7,139.7,'Tokyo'],
  [40.7,-74,'New York'],[34,-118.3,'LA'],[-23.5,-46.6,'Sao Paulo'],
  [51.5,-0.1,'London'],[48.9,2.4,'Paris'],[55.8,37.6,'Moscow'],
];
const hotGeo = new THREE.SphereGeometry(0.022, 8, 8);
hotspots.forEach(([lat,lon]) => {
  const colors = [0xff2200, 0xff5500, 0xff8800, 0xffaa00];
  const col = colors[Math.floor(Math.random()*colors.length)];
  const hotMat = new THREE.MeshBasicMaterial({ color:col });
  const phi   = (90-lat)*Math.PI/180;
  const theta = (lon+180)*Math.PI/180;
  const dot   = new THREE.Mesh(hotGeo, hotMat);
  dot.position.set(
    -Math.sin(phi)*Math.cos(theta),
     Math.cos(phi),
     Math.sin(phi)*Math.sin(theta)
  );
  earth.add(dot);
});

// Atmosphere glow
const atmosGeo = new THREE.SphereGeometry(1.08, 64, 64);
const atmosMat = new THREE.MeshPhongMaterial({
  color: 0x1a6aff,
  transparent: true,
  opacity: 0.08,
  side: THREE.BackSide,
});
scene.add(new THREE.Mesh(atmosGeo, atmosMat));

// Heat glow layer
const heatGeo = new THREE.SphereGeometry(1.04, 64, 64);
const heatMat = new THREE.MeshPhongMaterial({
  color: 0xff4400,
  transparent: true,
  opacity: 0.04,
  side: THREE.BackSide,
});
scene.add(new THREE.Mesh(heatGeo, heatMat));

// Lighting
scene.add(new THREE.AmbientLight(0x334466, 0.8));
const sun = new THREE.DirectionalLight(0xffffff, 1.2);
sun.position.set(5, 3, 5);
scene.add(sun);
const rimLight = new THREE.DirectionalLight(0x00d4aa, 0.4);
rimLight.position.set(-5, -2, -5);
scene.add(rimLight);

// Mouse interaction
let mouseX=0, mouseY=0;
document.addEventListener('mousemove', e => {
  mouseX = (e.clientX/window.innerWidth  - 0.5) * 0.5;
  mouseY = (e.clientY/window.innerHeight - 0.5) * 0.5;
});

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Animate
let t = 0;
function animate() {
  requestAnimationFrame(animate);
  t += 0.005;
  earth.rotation.y += 0.003;
  earth.rotation.x += (mouseY * 0.3 - earth.rotation.x) * 0.05;
  earth.rotation.y += (mouseX * 0.3) * 0.05;
  // Pulse heat glow
  heatMat.opacity = 0.03 + Math.sin(t*2) * 0.015;
  renderer.render(scene, camera);
}
animate();

// ── Particle System ───────────────────────────────────────────────────────────
const pc = document.getElementById('particles');
const ctx = pc.getContext('2d');
pc.width  = window.innerWidth;
pc.height = window.innerHeight;

const particles = [];
for(let i=0;i<80;i++) particles.push({
  x: Math.random()*pc.width,
  y: Math.random()*pc.height,
  r: Math.random()*2+0.5,
  vx: (Math.random()-0.5)*0.4,
  vy: -(Math.random()*0.5+0.2),
  life: Math.random(),
  col: Math.random() > 0.5 ? '255,75,75' : '0,212,170'
});

function drawParticles() {
  ctx.clearRect(0,0,pc.width,pc.height);
  particles.forEach(p => {
    p.x += p.vx; p.y += p.vy; p.life -= 0.003;
    if(p.life <= 0 || p.y < 0) {
      p.x=Math.random()*pc.width; p.y=pc.height;
      p.life=Math.random()*0.5+0.5;
    }
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.r, 0, Math.PI*2);
    ctx.fillStyle = `rgba(${p.col},${p.life*0.6})`;
    ctx.fill();
  });
  requestAnimationFrame(drawParticles);
}
drawParticles();

// ── Launch button ─────────────────────────────────────────────────────────────
function launch() {
  // Signal Streamlit via URL hash and auto-click a hidden element
  window.parent.postMessage({type:'streamlit:setComponentValue', value:true}, '*');
}
</script>
</body>
</html>
    """, height=700, scrolling=False)

    # Centered Streamlit launch button below the 3D scene
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2,1,2])
    with col2:
        if st.button("🚀 Launch Dashboard", type="primary", use_container_width=True):
            st.session_state["show_landing"] = False
            st.rerun()


st.set_page_config(
    page_title="Urban Heat Stress AI System",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get Help"   : "https://github.com/AS-youKnow/urban-heat-stress",
        "About"      : "AI/ML Urban Heat Stress Hotspot Prediction System — ISRO Project 2025-26",
    },
)


# ─── GLOBAL CSS & THEME ───────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary    : #0e1117;
    --bg-secondary  : #161b22;
    --bg-card       : #1c2333;
    --accent-red    : #ff4b4b;
    --accent-orange : #ff6b35;
    --accent-teal   : #00d4aa;
    --accent-blue   : #4a9eff;
    --text-primary  : #e6edf3;
    --text-secondary: #8b949e;
    --border-color  : #30363d;
    --gradient-hot  : linear-gradient(135deg, #ff4b4b, #ff8c00);
    --gradient-cool : linear-gradient(135deg, #00d4aa, #4a9eff);
}

/* ── App Root ── */
html, body, .stApp {
    background-color: var(--bg-primary);
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid var(--border-color);
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--accent-teal);
}

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #1a0a00 0%, #0d1117 40%, #001a1a 100%);
    border: 1px solid #ff6b3530;
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, #ff4b4b15 0%, transparent 60%),
                radial-gradient(circle at 70% 50%, #00d4aa10 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #ff6b35, #ff4b4b, #ffcc02);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 1rem;
    color: var(--text-secondary);
    margin-top: 8px;
}

/* ── Metric Cards ── */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: var(--accent-teal);
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent-teal);
}
.metric-label {
    font-size: 0.8rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

/* ── Status Chips ── */
.chip {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}
.chip-hot    { background:#ff4b4b20; color:#ff4b4b; border:1px solid #ff4b4b60; }
.chip-warn   { background:#ff8c0020; color:#ff8c00; border:1px solid #ff8c0060; }
.chip-cool   { background:#00d4aa20; color:#00d4aa; border:1px solid #00d4aa60; }
.chip-blue   { background:#4a9eff20; color:#4a9eff; border:1px solid #4a9eff60; }

/* ── Section Headers ── */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
    border-left: 3px solid var(--accent-teal);
    padding-left: 12px;
    margin: 20px 0 12px;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--text-secondary);
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.9rem;
}
.stTabs [aria-selected="true"] {
    background: var(--bg-card) !important;
    color: var(--accent-teal) !important;
}

/* ── DataFrame ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent-teal) !important; }

/* ── Slider ── */
.stSlider [data-baseweb="slider"] { color: var(--accent-teal); }

/* ── Warning / Info boxes ── */
.stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─── PLOTLY DARK THEME CONFIG ─────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#161b22",
    plot_bgcolor ="#0e1117",
    font_color   ="#e6edf3",
    font_family  ="Inter",
    margin       =dict(l=40, r=40, t=50, b=40),
    colorway     =["#ff4b4b", "#00d4aa", "#4a9eff", "#ffcc02", "#ff6b35"],
)

# ─── HOTSPOT COLOUR MAP ───────────────────────────────────────────────────────
HOTSPOT_COLORS = {
    "Extreme Hotspot (99%)": "#ff1a1a",
    "Hotspot (95%)"        : "#ff7f00",
    "Neutral"              : "#4a9eff",
    "Coldspot"             : "#00d4aa",
}


# ─── CACHE HELPERS ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def cached_pipeline(use_gee: bool, budget_n: int,
                    alpha: float, beta: float,
                    region_name: str = "Delhi, India") -> dict:
    """
    Run the complete ML pipeline and cache the result.
    Re-runs automatically when any input parameter changes.
    """
    from config import WORLD_REGIONS
    bbox = WORLD_REGIONS.get(region_name, CFG.roi_bbox)

    # Module 1 — Ingestion
    df = run_ingestion(use_gee=use_gee, roi_bbox=bbox, region_name=region_name)

    # Module 2 — Clustering
    df = run_hotspot_analysis(df)

    # Module 3 — Modeling
    mod_results = run_modeling(df)

    # Module 4 — SHAP
    shap_results = run_shap_analysis(mod_results)

    # Module 5 — Simulation
    sim_results = run_simulation(mod_results)

    # Module 6 — Optimization
    opt_results = run_full_optimization(
        simulation_results=sim_results,
        modeling_results=mod_results,
        budget_n=budget_n,
        alpha=alpha,
        beta=beta,
    )

    return {
        "df"           : df,
        "mod_results"  : mod_results,
        "shap_results" : shap_results,
        "sim_results"  : sim_results,
        "opt_results"  : opt_results,
        "region_name"  : region_name,
    }


# ─── MAP BUILDER ──────────────────────────────────────────────────────────────

def build_hotspot_map(df: pd.DataFrame,
                       top_n_df: pd.DataFrame,
                       show_heatmap: bool = True) -> folium.Map:
    """
    Construct an interactive Folium map with:
      • Choropleth circle markers coloured by hotspot category
      • Optional LST heat map overlay
      • Red star markers for top-N optimization targets

    Parameters
    ----------
    df        : pd.DataFrame — full grid with hotspot_category
    top_n_df  : pd.DataFrame — target cells from Module 6
    show_heatmap : bool      — overlay the LST heat map layer

    Returns
    -------
    folium.Map
    """
    # Centre on median coordinates
    centre_lat = df["latitude"].median()
    centre_lon = df["longitude"].median()

    fmap = folium.Map(
        location=[centre_lat, centre_lon],
        zoom_start=11,
        tiles="CartoDB dark_matter",
        prefer_canvas=True,
    )

    # ── Hotspot markers ────────────────────────────────────────────────────────
    # Sample for display performance (max 1500 points)
    display_df = df.sample(min(1500, len(df)), random_state=42)

    for _, row in display_df.iterrows():
        color = HOTSPOT_COLORS.get(row.get("hotspot_category", "Neutral"), "#4a9eff")
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            weight=0,
            popup=folium.Popup(
                f"<b>LST:</b> {row.get('LST_Celsius', 'N/A'):.2f} °C<br>"
                f"<b>Category:</b> {row.get('hotspot_category', 'N/A')}<br>"
                f"<b>NDVI:</b> {row.get('NDVI', 'N/A'):.3f}<br>"
                f"<b>NDBI:</b> {row.get('NDBI', 'N/A'):.3f}",
                max_width=200,
            ),
        ).add_to(fmap)

    # ── LST HeatMap overlay ────────────────────────────────────────────────────
    if show_heatmap and "LST_Celsius" in df.columns:
        heat_data = df[["latitude", "longitude", "LST_Celsius"]].dropna()
        # Normalise LST for heat map intensity
        lst_min = heat_data["LST_Celsius"].min()
        lst_max = heat_data["LST_Celsius"].max()
        heat_data = heat_data.copy()
        heat_data["intensity"] = (
            (heat_data["LST_Celsius"] - lst_min) / (lst_max - lst_min + 1e-6)
        )
        HeatMap(
            heat_data[["latitude", "longitude", "intensity"]].values.tolist(),
            name="LST Heat Map",
            radius=12,
            blur=10,
            max_zoom=13,
            gradient={0.0: "blue", 0.4: "lime", 0.65: "yellow", 1.0: "red"},
        ).add_to(fmap)

    # ── Target cell markers ────────────────────────────────────────────────────
    if top_n_df is not None and len(top_n_df) > 0:
        target_group = folium.FeatureGroup(name="🎯 Optimization Targets")
        for _, row in top_n_df.head(100).iterrows():
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                icon=folium.DivIcon(
                    html=f"""
                    <div style="
                        width:16px; height:16px;
                        border-radius:50%;
                        background:radial-gradient(circle, #ff0000, #800000);
                        border:2px solid #ffffff;
                        box-shadow:0 0 8px #ff0000;
                    "></div>
                    """,
                    icon_size=(16, 16),
                    icon_anchor=(8, 8),
                ),
                popup=folium.Popup(
                    f"<b>🎯 Target #{int(row.get('Rank', 0))}</b><br>"
                    f"<b>LST:</b> {row.get('LST_Celsius', 0):.2f}°C<br>"
                    f"<b>Strategy:</b> {row.get('Best_Scenario', 'N/A')}<br>"
                    f"<b>Expected ΔT:</b> {row.get('Best_DeltaT_C', 0):+.3f}°C<br>"
                    f"<b>Priority:</b> {row.get('Priority_Score', 0):.3f}",
                    max_width=220,
                ),
            ).add_to(target_group)
        target_group.add_to(fmap)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:9999;
                background:#1c2333ee; border:1px solid #30363d;
                border-radius:10px; padding:14px 18px; font-family:Inter,sans-serif;">
      <b style="color:#e6edf3; font-size:13px;">Heat Category</b><br>
      <span style="color:#ff1a1a">●</span>
        <span style="color:#e6edf3; font-size:12px;"> Extreme Hotspot (99%)</span><br>
      <span style="color:#ff7f00">●</span>
        <span style="color:#e6edf3; font-size:12px;"> Hotspot (95%)</span><br>
      <span style="color:#4a9eff">●</span>
        <span style="color:#e6edf3; font-size:12px;"> Neutral</span><br>
      <span style="color:#00d4aa">●</span>
        <span style="color:#e6edf3; font-size:12px;"> Coldspot</span><br>
      <span style="color:#ff0000">⬤</span>
        <span style="color:#e6edf3; font-size:12px;"> 🎯 Target Cell</span>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(fmap)

    return fmap


# ─── CHART BUILDERS ───────────────────────────────────────────────────────────

def plot_actual_vs_predicted(y_test, y_pred) -> go.Figure:
    """Scatter plot of actual vs. predicted LST with identity line."""
    fig = px.scatter(
        x=y_test, y=y_pred,
        labels={"x": "Actual LST (°C)", "y": "Predicted LST (°C)"},
        title="Actual vs. Predicted Land Surface Temperature",
        opacity=0.55,
        color_discrete_sequence=["#00d4aa"],
    )
    mn = min(float(y_test.min()), float(y_pred.min()))
    mx = max(float(y_test.max()), float(y_pred.max()))
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx],
        mode="lines",
        line=dict(color="#ff4b4b", dash="dash", width=2),
        name="Perfect Prediction",
        showlegend=True,
    ))
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def plot_residual_histogram(y_test, y_pred) -> go.Figure:
    """Distribution of residuals (actual − predicted)."""
    residuals = np.array(y_test) - np.array(y_pred)
    fig = px.histogram(
        x=residuals, nbins=60,
        labels={"x": "Residual (°C)", "y": "Count"},
        title="Residual Distribution",
        color_discrete_sequence=["#4a9eff"],
    )
    fig.add_vline(x=0, line_color="#ff4b4b", line_dash="dash", line_width=2)
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def plot_scenario_comparison(summary: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing scenario metrics."""
    metrics   = ["Mean_DeltaT_C", "Median_DeltaT_C", "P95_DeltaT_C", "Max_DeltaT_C"]
    labels    = ["Mean ΔT", "Median ΔT", "95th Pct ΔT", "Max ΔT"]
    colors    = ["#ff4b4b", "#ff6b35", "#ffcc02", "#00d4aa"]
    scenarios = summary["Scenario"].tolist()

    fig = go.Figure()
    for metric, label, color in zip(metrics, labels, colors):
        fig.add_trace(go.Bar(
            name=label,
            x=scenarios,
            y=summary[metric],
            marker_color=color,
            text=[f"{v:+.3f}°C" for v in summary[metric]],
            textposition="outside",
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Scenario Cooling Comparison (ΔT in °C — higher = more cooling)",
        barmode="group",
        bargap=0.15,
        bargroupgap=0.05,
        yaxis_title="ΔT (°C Cooling)",
        xaxis_title="Intervention Scenario",
        legend=dict(bgcolor="#1c2333", bordercolor="#30363d", borderwidth=1),
    )
    return fig


def plot_cells_cooled(summary: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of % cells cooled per scenario."""
    fig = px.bar(
        summary.sort_values("Pct_Cells_Cooled"),
        x="Pct_Cells_Cooled",
        y="Scenario",
        orientation="h",
        text="Pct_Cells_Cooled",
        color="Pct_Cells_Cooled",
        color_continuous_scale=[[0, "#1e3a2f"], [0.5, "#00b890"], [1, "#00d4aa"]],
        title="Percentage of Grid Cells Experiencing Cooling",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False,
                      xaxis_title="% Cells Cooled", yaxis_title="")
    return fig


def plot_strategy_donut(strategy_report: pd.DataFrame) -> go.Figure:
    """Donut chart of strategy allocation across target cells."""
    fig = px.pie(
        strategy_report,
        names="Best_Scenario",
        values="Num_Cells",
        hole=0.55,
        title="Strategy Allocation Across Target Cells",
        color_discrete_sequence=["#ff4b4b", "#00d4aa", "#4a9eff", "#ffcc02"],
    )
    fig.update_traces(
        textinfo="label+percent",
        textfont_size=12,
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def plot_priority_scatter(full_opt_df: pd.DataFrame) -> go.Figure:
    """Scatter plot of Severity vs. Cooling Sensitivity, coloured by Priority."""
    fig = px.scatter(
        full_opt_df,
        x="Severity_Score",
        y="Sensitivity_Score",
        color="Priority_Score",
        color_continuous_scale="RdYlGn_r",
        title="Cell Priority Landscape — Severity vs. Cooling Sensitivity",
        labels={
            "Severity_Score"    : "Temperature Severity Score",
            "Sensitivity_Score" : "Cooling Sensitivity Score",
            "Priority_Score"    : "Priority",
        },
        opacity=0.5,
        hover_data={
            "LST_Celsius"      : True,
            "Best_Scenario"    : True,
            "Best_DeltaT_C"    : True,
        },
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def plot_hotspot_distribution(df: pd.DataFrame) -> go.Figure:
    """Donut chart of hotspot category distribution."""
    counts = df["hotspot_category"].value_counts().reset_index()
    counts.columns = ["Category", "Count"]
    colors = [HOTSPOT_COLORS.get(c, "#888") for c in counts["Category"]]
    fig = px.pie(
        counts, names="Category", values="Count", hole=0.55,
        title="Hotspot Category Distribution",
        color="Category",
        color_discrete_map=HOTSPOT_COLORS,
    )
    fig.update_traces(textinfo="label+percent")
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# ─── SHAP PNG DISPLAY HELPERS ─────────────────────────────────────────────────

def display_shap_png(path: str, caption: str = "") -> None:
    """Load and display a SHAP PNG with a styled caption."""
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"SHAP plot not found at `{path}`. Run the pipeline first.")


# ─── HEADER COMPONENT ─────────────────────────────────────────────────────────

def render_header() -> None:
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">🌡️ Urban Heat Stress AI System</h1>
        <p class="hero-subtitle">
            AI/ML-powered hotspot prediction, simulation &amp; optimization
            for sustainable urban planning across India
        </p>
        <div style="margin-top:14px; display:flex; gap:10px; flex-wrap:wrap;">
            <span class="chip chip-hot">🔴 LST Prediction</span>
            <span class="chip chip-warn">🟠 Getis-Ord Gi* Clustering</span>
            <span class="chip chip-blue">🔵 XGBoost + SHAP</span>
            <span class="chip chip-cool">🟢 Scenario Simulation</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """Render sidebar controls and return settings dict."""
    with st.sidebar:
        st.markdown("## ⚙️ Pipeline Controls")
        st.divider()

        # ── Data source ───────────────────────────────────────────────────────
        data_source = st.radio(
            "📡 Data Source",
            options=["🌐 Google Earth Engine (Live)", "🧪 Synthetic Demo"],
            index=1,
            help="GEE requires `earthengine authenticate` and a project ID in config.py",
        )
        use_gee = data_source.startswith("🌐")

        if use_gee:
            st.info("⚠️ GEE auth required. Set `GEE_PROJECT_ID` in `config.py`.",
                    icon="ℹ️")

        st.divider()

        # ── World Region Selector ─────────────────────────────────────────────
        st.markdown("### 🌍 Select City / Region")

        # Group cities by continent for the selectbox
        from config import WORLD_REGIONS
        all_regions = list(WORLD_REGIONS.keys())

        # Continent headers as disabled separators
        continents = {
            "── ASIA ──────────────": [
                r for r in all_regions
                if any(c in r for c in [", India","Japan","China","Pakistan",
                    "Bangladesh","Thailand","Indonesia","Philippines",
                    "Singapore","Korea","Saudi","UAE","Iran","Afghanistan",
                    "Sri Lanka"])
            ],
            "── AFRICA ────────────": [
                r for r in all_regions
                if any(c in r for c in ["Egypt","Nigeria","Kenya","Africa",
                    "Sudan","Congo","Ethiopia","Ghana"])
            ],
            "── EUROPE ────────────": [
                r for r in all_regions
                if any(c in r for c in ["UK","France","Spain","Italy",
                    "Greece","Turkey","Russia"])
            ],
            "── AMERICAS ──────────": [
                r for r in all_regions
                if any(c in r for c in ["USA","Mexico","Brazil",
                    "Argentina","Colombia","Peru"])
            ],
            "── AUSTRALIA ─────────": [
                r for r in all_regions
                if "Australia" in r
            ],
        }

        # Build flat ordered list with continent headers
        ordered = []
        for header, cities in continents.items():
            ordered.append(header)
            ordered.extend(cities)

        selected_raw = st.selectbox(
            "City",
            options=ordered,
            index=ordered.index("Delhi, India") if "Delhi, India" in ordered else 1,
            help="Select any city — the map, data, and analysis will shift to that region",
            label_visibility="collapsed",
        )

        # Skip if user selected a continent header
        region_name = selected_raw if selected_raw in WORLD_REGIONS else "Delhi, India"

        # Show selected city bbox
        bbox = WORLD_REGIONS[region_name]
        st.markdown(
            f"""
            <div style='background:#1c2333; border:1px solid #30363d;
                        border-radius:8px; padding:10px 14px; margin-top:6px;
                        font-size:0.8rem; color:#8b949e;'>
            <b style='color:#00d4aa;'>📍 {region_name}</b><br>
            Lat: {bbox[1]:.2f}° – {bbox[3]:.2f}°<br>
            Lon: {bbox[0]:.2f}° – {bbox[2]:.2f}°
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown("### 🎯 Optimization Budget")
        budget_n = st.slider(
            "Number of target grid cells (N)",
            min_value=10, max_value=500, value=CFG.budget_n, step=10,
            help="How many high-priority cells to include in the intervention plan",
        )

        st.divider()
        st.markdown("### ⚖️ Optimization Weights")
        alpha_val = st.slider(
            "Severity Weight",
            min_value=0.0, max_value=1.0, value=float(CFG.alpha), step=0.05,
            help="Higher value prioritises cells with extreme baseline temperatures",
        )
        beta_val = 1.0 - alpha_val
        st.markdown(
            f"<p style='font-size:0.85rem; color:#8b949e;'>"
            f"Sensitivity Weight: <b style='color:#00d4aa'>{beta_val:.2f}</b>"
            f" (auto = 1 - alpha)</p>",
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown("### 🗺️ Map Options")
        show_heatmap = st.checkbox("Show LST Heat Map Overlay", value=True)

        st.divider()
        run_btn = st.button(
            "🚀 Run Pipeline",
            use_container_width=True,
            type="primary",
        )

        st.divider()
        st.markdown(
            "<p style='font-size:0.75rem; color:#8b949e; text-align:center;'>"
            "Urban Heat Stress AI System<br>ISRO Research Project 2025-26</p>",
            unsafe_allow_html=True,
        )

    return {
        "use_gee"      : use_gee,
        "region_name"  : region_name,
        "budget_n"     : budget_n,
        "alpha"        : alpha_val,
        "beta"         : beta_val,
        "show_heatmap" : show_heatmap,
        "run_pipeline" : run_btn,
    }


# ─── TAB RENDERERS ────────────────────────────────────────────────────────────

def render_tab_map(df: pd.DataFrame, top_n_df: pd.DataFrame,
                    show_heatmap: bool) -> None:
    """Tab 1: Interactive heat map."""
    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.markdown('<p class="section-header">🗺️ Urban Heat Hotspot Map</p>',
                    unsafe_allow_html=True)
        fmap = build_hotspot_map(df, top_n_df, show_heatmap)
        st_folium(fmap, width=None, height=520, returned_objects=[])

    with col_b:
        st.markdown('<p class="section-header">📊 Category Distribution</p>',
                    unsafe_allow_html=True)
        st.plotly_chart(
            plot_hotspot_distribution(df),
            use_container_width=True,
        )

        st.markdown('<p class="section-header">🌡️ Vulnerability Profile</p>',
                    unsafe_allow_html=True)
        profile = heat_vulnerability_profile(df)
        st.dataframe(
            profile.style.format("{:.3f}").background_gradient(
                cmap="RdYlGn_r", subset=["LST_Celsius"]
            ),
            use_container_width=True,
        )


def render_tab_model(mod_results: dict) -> None:
    """Tab 2: Model performance metrics and charts."""
    metrics = mod_results["metrics"]
    y_test  = mod_results["y_test"]
    y_pred  = metrics["y_pred"]

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("MAE", f"{metrics['MAE']:.3f} °C", "Mean Absolute Error"),
        ("RMSE", f"{metrics['RMSE']:.3f} °C", "Root Mean Square Error"),
        ("R²", f"{metrics['R2']:.4f}", "Coefficient of Determination"),
        ("Train N", f"{len(mod_results['X_train']):,}", "Training Samples"),
    ]
    for col, (label, value, description) in zip([c1, c2, c3, c4], kpis):
        with col:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                    <div style="font-size:0.72rem;color:#8b949e;margin-top:4px;">
                        {description}
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    c_left, c_right = st.columns(2)
    with c_left:
        st.plotly_chart(
            plot_actual_vs_predicted(y_test, y_pred),
            use_container_width=True,
        )
    with c_right:
        st.plotly_chart(
            plot_residual_histogram(y_test, y_pred),
            use_container_width=True,
        )

    # ── Feature importances (native XGBoost) ──────────────────────────────────
    st.markdown('<p class="section-header">🌲 XGBoost Native Feature Importances (Gain)</p>',
                unsafe_allow_html=True)
    fi_vals = mod_results["model"].feature_importances_
    fi_df = pd.DataFrame({
        "Feature": CFG.feature_cols,
        "Importance (Gain)": fi_vals,
    }).sort_values("Importance (Gain)", ascending=False)

    fig_fi = px.bar(
        fi_df, x="Feature", y="Importance (Gain)",
        color="Importance (Gain)",
        color_continuous_scale="RdYlGn_r",
        title="XGBoost Feature Importance by Gain",
    )
    fig_fi.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False)
    st.plotly_chart(fig_fi, use_container_width=True)


def render_tab_shap(shap_results: dict) -> None:
    """Tab 3: SHAP driver analysis."""
    importance_df = shap_results["importance_df"]

    # ── SHAP Importance Table ─────────────────────────────────────────────────
    st.markdown('<p class="section-header">🔬 SHAP Feature Importance Summary</p>',
                unsafe_allow_html=True)

    cols = st.columns([1, 1])
    with cols[0]:
        # Interactive bar chart from SHAP importance data
        fig_shap = px.bar(
            importance_df.sort_values("Mean_Abs_SHAP"),
            x="Mean_Abs_SHAP",
            y="Feature",
            orientation="h",
            color="Mean_Abs_SHAP",
            color_continuous_scale="RdYlGn_r",
            title="Mean |SHAP| — Average Impact on LST Prediction (°C)",
            text=importance_df.sort_values("Mean_Abs_SHAP")["Mean_Abs_SHAP"].round(3),
        )
        fig_shap.update_traces(texttemplate="%{text} °C", textposition="outside")
        fig_shap.update_layout(
            **PLOTLY_LAYOUT,
            coloraxis_showscale=False,
            xaxis_title="Mean |SHAP Value| (°C)",
        )
        st.plotly_chart(fig_shap, use_container_width=True)

    with cols[1]:
        st.markdown("""
        <div style="background:#1c2333; border:1px solid #30363d; border-radius:12px;
                    padding:20px; height:100%;">
        <b style="color:#00d4aa; font-size:1rem;">Interpreting SHAP Values</b><br><br>
        <p style="color:#c9d1d9; font-size:0.9rem; line-height:1.7;">
        SHAP (SHapley Additive exPlanations) quantifies how much each feature
        contributes to pushing the predicted LST above or below the baseline average.
        <br><br>
        <b style="color:#ff4b4b;">Positive SHAP</b> → Feature is associated with
        <b>higher LST</b> (surface heating)<br>
        <b style="color:#00d4aa;">Negative SHAP</b> → Feature drives the prediction
        <b>toward cooler temperatures</b><br><br>
        <b>Key Insights:</b><br>
        • High <b>NDBI</b> / <b>Built_Fraction</b> → dominant heat sources<br>
        • High <b>NDVI</b> → negative SHAP (green areas cool the surface)<br>
        • <b>Pop_Density</b> → indirect urban heat island contribution<br>
        • <b>LULC</b> = 50 (built-up) → strongest heating signal
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── SHAP PNG Plots ─────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">📈 Global SHAP Beeswarm Plot</p>',
                unsafe_allow_html=True)
    display_shap_png(
        shap_results["beeswarm_path"],
        "Each dot = one test cell. X-position = SHAP value (°C impact). "
        "Colour = feature value (red=high, blue=low)."
    )

    st.markdown('<p class="section-header">📊 SHAP Feature Importance Bar</p>',
                unsafe_allow_html=True)
    display_shap_png(
        shap_results["bar_path"],
        "Mean absolute SHAP value per feature — higher = more influential."
    )


def render_tab_scenarios(sim_results: dict) -> None:
    """Tab 4: Scenario simulation comparison."""
    summary    = sim_results["summary"]
    results_df = sim_results["results_df"]

    # ── Metric overview cards ─────────────────────────────────────────────────
    st.markdown('<p class="section-header">📉 Scenario Cooling Summary</p>',
                unsafe_allow_html=True)
    cols = st.columns(len(summary))
    for col, (_, row) in zip(cols, summary.iterrows()):
        with col:
            st.markdown(
                f"""<div class="metric-card">
                    <div style="font-size:0.85rem; color:#8b949e; margin-bottom:6px;">
                        {row['Scenario']}</div>
                    <div class="metric-value" style="font-size:1.6rem;">
                        {row['Mean_DeltaT_C']:+.3f}°C</div>
                    <div class="metric-label">Mean Cooling</div>
                    <div style="margin-top:8px; font-size:0.78rem; color:#8b949e;">
                        {row['Pct_Cells_Cooled']:.1f}% cells cooled
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Scenario comparison bar chart ─────────────────────────────────────────
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(
            plot_scenario_comparison(summary),
            use_container_width=True,
        )
    with col_right:
        st.plotly_chart(
            plot_cells_cooled(summary),
            use_container_width=True,
        )

    # ── ΔT distribution violin plots ──────────────────────────────────────────
    st.markdown('<p class="section-header">🎻 Cooling Effect Distribution per Scenario</p>',
                unsafe_allow_html=True)

    delta_cols = [c for c in results_df.columns if c.startswith("DeltaT_")]
    violin_data = []
    for col in delta_cols:
        name = col.replace("DeltaT_", "").replace("_", " ")
        for val in results_df[col].values:
            violin_data.append({"Scenario": name, "ΔT (°C)": val})
    violin_df = pd.DataFrame(violin_data)

    fig_violin = px.violin(
        violin_df,
        x="Scenario",
        y="ΔT (°C)",
        color="Scenario",
        box=True,
        points=False,
        title="Distribution of Cooling Effect Across All Grid Cells",
        color_discrete_sequence=["#ff4b4b", "#00d4aa", "#4a9eff"],
    )
    fig_violin.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_violin, use_container_width=True)

    # ── Raw summary table ─────────────────────────────────────────────────────
    st.markdown('<p class="section-header">📋 Full Simulation Metrics Table</p>',
                unsafe_allow_html=True)
    display_summary = summary.copy()
    display_summary["Mean_DeltaT_C"]   = display_summary["Mean_DeltaT_C"].map("{:+.3f}°C".format)
    display_summary["Median_DeltaT_C"] = display_summary["Median_DeltaT_C"].map("{:+.3f}°C".format)
    display_summary["P95_DeltaT_C"]    = display_summary["P95_DeltaT_C"].map("{:+.3f}°C".format)
    display_summary["Max_DeltaT_C"]    = display_summary["Max_DeltaT_C"].map("{:+.3f}°C".format)
    display_summary["Pct_Cells_Cooled"]= display_summary["Pct_Cells_Cooled"].map("{:.1f}%".format)
    st.dataframe(display_summary.set_index("Scenario"), use_container_width=True)


def render_tab_optimization(opt_results: dict, budget_n: int) -> None:
    """Tab 5: Optimization results."""
    top_n_df        = opt_results["top_n_df"]
    full_opt_df     = opt_results["full_opt_df"]
    strategy_report = opt_results["strategy_report"]

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("Target Cells", str(len(top_n_df)), "Selected for intervention"),
        ("Avg LST", f"{top_n_df['LST_Celsius'].mean():.2f}°C", "In target cells"),
        ("Avg Priority", f"{top_n_df['Priority_Score'].mean():.3f}", "Priority score"),
        ("Avg Cooling", f"{top_n_df['Best_DeltaT_C'].mean():+.3f}°C",
         "Expected ΔT per cell"),
    ]
    for col, (label, value, desc) in zip([c1, c2, c3, c4], kpis):
        with col:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                    <div style="font-size:0.72rem;color:#8b949e;margin-top:4px;">
                        {desc}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(plot_strategy_donut(strategy_report), use_container_width=True)
    with col_b:
        st.plotly_chart(plot_priority_scatter(full_opt_df), use_container_width=True)

    # ── Strategy allocation table ──────────────────────────────────────────────
    st.markdown('<p class="section-header">📋 Strategy Allocation Report</p>',
                unsafe_allow_html=True)
    st.dataframe(
        strategy_report.style.format({
            "Avg_DeltaT_C" : "{:+.3f}°C",
            "Avg_Priority" : "{:.3f}",
            "Avg_LST_C"    : "{:.2f}°C",
        }),
        use_container_width=True,
    )

    # ── Top-N target cells table ───────────────────────────────────────────────
    st.markdown(
        f'<p class="section-header">🎯 Top-{len(top_n_df)} Target Cells</p>',
        unsafe_allow_html=True,
    )
    display_top = top_n_df[[
        "Rank", "longitude", "latitude", "LST_Celsius",
        "Priority_Score", "Best_Scenario", "Best_DeltaT_C",
    ]].copy()
    display_top["LST_Celsius"]    = display_top["LST_Celsius"].map("{:.2f}°C".format)
    display_top["Priority_Score"] = display_top["Priority_Score"].map("{:.3f}".format)
    display_top["Best_DeltaT_C"]  = display_top["Best_DeltaT_C"].map("{:+.3f}°C".format)
    display_top["longitude"]      = display_top["longitude"].round(5)
    display_top["latitude"]       = display_top["latitude"].round(5)

    st.dataframe(display_top, use_container_width=True, height=400)

    # ── Download button ───────────────────────────────────────────────────────
    csv_bytes = top_n_df.to_csv(index=False).encode()
    st.download_button(
        label     = "📥 Download Target Cells CSV",
        data      = csv_bytes,
        file_name = "optimization_targets.csv",
        mime      = "text/csv",
        type      = "primary",
    )


# ─── MAIN APP ─────────────────────────────────────────────────────────────────

def main():
    render_header()
    settings = render_sidebar()

    # ── Pipeline execution ────────────────────────────────────────────────────
    if "pipeline_data" not in st.session_state or settings["run_pipeline"]:
        with st.spinner("Running AI/ML pipeline for " + settings.get("region_name", "selected city") + " — please wait..."):
            try:
                data = cached_pipeline(
                    use_gee     = settings["use_gee"],
                    budget_n    = settings["budget_n"],
                    alpha       = settings["alpha"],
                    beta        = settings["beta"],
                    region_name = settings["region_name"],
                )
                st.session_state["pipeline_data"] = data
                st.session_state["settings"]      = settings
                st.success("✅ Pipeline complete! Explore the tabs below.", icon="🎉")
            except Exception as exc:
                st.error(f"❌ Pipeline failed: {exc}", icon="🚨")
                st.exception(exc)
                st.stop()

    # ── Retrieve cached results ───────────────────────────────────────────────
    if "pipeline_data" not in st.session_state:
        st.info("👈 Click **Run Pipeline** in the sidebar to begin.", icon="ℹ️")
        st.stop()

    data     = st.session_state["pipeline_data"]
    df       = data["df"]
    top_n_df = data["opt_results"]["top_n_df"]

    # ── Overview ribbon ───────────────────────────────────────────────────────
    o1, o2, o3, o4, o5 = st.columns(5)
    source_label = df.attrs.get("source", "unknown").upper()
    cat_counts   = df["hotspot_category"].value_counts()
    hot_count    = cat_counts.get("Extreme Hotspot (99%)", 0) + cat_counts.get("Hotspot (95%)", 0)

    for col, (label, value) in zip(
        [o1, o2, o3, o4, o5],
        [
            ("Grid Cells",      f"{len(df):,}"),
            ("Data Source",     source_label),
            ("Hotspot Cells",   f"{hot_count:,}"),
            ("Mean LST",        f"{df['LST_Celsius'].mean():.2f}°C"),
            ("Max LST",         f"{df['LST_Celsius'].max():.2f}°C"),
        ],
    ):
        with col:
            st.markdown(
                f"""<div class="metric-card" style="padding:14px;">
                    <div class="metric-value" style="font-size:1.4rem;">{value}</div>
                    <div class="metric-label" style="font-size:0.7rem;">{label}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️  Heat Map",
        "📈  Model Performance",
        "🔬  SHAP Analysis",
        "🔄  Scenarios",
        "🎯  Optimization",
    ])

    with tab1:
        render_tab_map(df, top_n_df, settings["show_heatmap"])

    with tab2:
        render_tab_model(data["mod_results"])

    with tab3:
        render_tab_shap(data["shap_results"])

    with tab4:
        render_tab_scenarios(data["sim_results"])

    with tab5:
        render_tab_optimization(data["opt_results"], settings["budget_n"])


if __name__ == "__main__":
    # Show landing page on first visit, then the main dashboard
    if "show_landing" not in st.session_state:
        st.session_state["show_landing"] = True

    if st.session_state["show_landing"]:
        render_landing_page()
    else:
        main()
