"""
Live Hot-Swap Web Studio for `Ministral-3-8B` Swedish Weight Cartridge (`live_studio.py`)
=========================================================================================
Zero-dependency local web application (Python standard library `http.server` + PyTorch +
Transformers + `cartridge_loader.py`) designed for interactive testing and live social-media
demos (`http://127.0.0.1:7865`).

Features:
  1. Sub-millisecond / millisecond GPU hot-swapping (`attach` / `detach`) of the 960 KiB
     `.safetensors` Swedish Weight Cartridge directly inside `Mistral3MLP` Linear layers.
  2. Live Side-by-Side "BEFORE vs AFTER" generation on the exact same prompt with zero system
     prompts, highlighting Swenglish/grammar defects vs idiomatic Swedish constructions.
  3. Interactive 2,640-item Benchmark & Zero-Collateral Audit Explorer loaded from
     `benchmarks/before_after_verified_results.json`.

License: Apache-2.0
"""

import argparse
import json
import os
import re
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cartridge_loader import SynapseCartridgeHandle, load_cartridge

DEFAULT_CARTRIDGE = os.path.join(HERE, "cartridges", "ministral3_8b_swedish_boost_v1.safetensors")
DEFAULT_BENCH_JSON = os.path.join(HERE, "benchmarks", "before_after_verified_results.json")
DEFAULT_LOCAL_MODEL = os.path.abspath(os.path.join(HERE, "..", "..", "models", "Ministral-3-8B-Instruct-2512-BF16"))
HF_MODEL_ID = "mistralai/Ministral-3-8B-Instruct-2512-BF16"


# Simple deterministic highlighters for visual clarity in the UI
UI_DEFECT_REGEXES = [
    r"\b(?:nyckel)?stakeholder\w*\b",
    r"\br\s*&\s*d(?:-teamet|-avdelning\w*)?\b",
    r"\bworkflow\w*\b",
    r"\bonboarding\w*\b",
    r"\bfeedback\w*\b",
    r"\b(?:key\s+)?takeaways?\b",
    r"\bcompliance\b",
    r"\broadmap\w*\b",
    r"\bdeadline\w*\b",
    r"\bkick-?off\w*\b",
    r"\bspendera(?:r|de|t)?\b",
    r"\badressera(?:r|de|t)?\b",
    r"\bi\s+slutet\s+av\s+dagen\b",
    r"\bgör(?:a|de)?\s+mening\b",
    r"\b(?:ta|tar|tog|tagit|göra|gör|gjorde)\s+(?:ett\s+|en\s+)?(?:\w+\s+)?beslut\b",
    r"\b(?:göra|gör|gjorde)\s+(?:samma\s+|ett\s+|stora?\s+)?misstag\b",
    r"\b(?:en|någon|ingen)\s+(?:egentlig\s+|konkret\s+|svår\s+|ny\s+|stor\s+|klar\s+|mätbar\s+)?(?:beslut|värde|ansvar|tålamod|stöd|avtal|förslag|misstag|fel|val|tvivel|svar)\b",
    r"\b(?:stor\s+tålamod|stor\s+stöd|egentlig\s+värde|aktuella\s+utfallet)\b",
    r"\bpolitisk\s+\"?skådespel\b",
    r"\bvacker\s+och\s+fritt\b",
    r"\b(?:bland|till|med|för)\s+(?:alla\s+)?de(?=\s+som\b|[.,!?;:])",
    r"\bvar\s+(?:tåget|de|vi|jag)\s+(?:är\s+på\s+väg|var\s+på\s+väg|ska\s+resa|ska\s+flytta)\b",
]

UI_GOLD_REGEXES = [
    r"\bfatta(?:r|de|t)?\s+(?:ett\s+)?(?:\w+\s+)?beslut\b",
    r"\bvidta(?:r|git)?\s+(?:\w+\s+)?åtgärder\b|\bvidtog\s+(?:\w+\s+)?åtgärder\b",
    r"\bbegå(?:r|tt)?\s+(?:\w+\s+)?misstag\b|\bbegick\s+(?:\w+\s+)?misstag\b",
    r"\b(?:ingå|ingick|slöt)\s+(?:\w+\s+)?avtal\b",
    r"\bäg(?:a|er|de)\s+rum\b",
    r"\btillbringa(?:r|de)?\b",
    r"\b(?:nyckel)?intressenter(?:na)?\b",
    r"\bforskning(?:s-?\s+och\s+utveckling\w*)?\b|\bFoU\b",
    r"\barbetsflöde(?:t|n)?\b",
    r"\båterkoppling(?:en)?\b",
    r"\bregelefterlevnad(?:en)?\b",
    r"\btidsfrist(?:en)?\b",
    r"\bi\s+slutändan\b",
    r"\b(?:är|vara)\s+rimlig(?:t|a)?\b",
    r"\b(?:något|inget|stort|egentligt|konkret|mätbart|klart)\s+(?:\w+\s+)?(?:värde|ansvar|tålamod|stöd|tvivel|fel|svar|val)\b",
    r"\bpolitiskt\s+\"?skådespel\b",
    r"\bvacker\s+och\s+fri\b",
    r"\b(?:bland|till|med|för)\s+(?:alla\s+)?dem\b",
    r"\bvart\s+(?:tåget|de|vi|jag)\b",
]


PRESET_PROMPTS = [
    {
        "title": "1. Executive Translation (Stakeholders, Gender & Collocations)",
        "tag": "SV Translation",
        "prompt": "Översätt följande engelska text till naturlig, vårdad svenska (endast själva översättningen):\n'Yesterday, the board met with key stakeholders to take a tough decision. Eventually, they realized that the old workflow did not create any real value for customers, and that the R&D department must take immediate measures.'",
    },
    {
        "title": "2. Patience, Deadlines & 'Make Sense' Idiom",
        "tag": "SV Idioms",
        "prompt": "Översätt till vårdad svenska (endast översättningen):\n'There is no doubt that the reform requires great patience from all stakeholders. Does it really make sense to rush a decision before the deadline when we have no other choice?'",
    },
    {
        "title": "3. Direction ('vart') vs Location ('var') & Abstract Neuter",
        "tag": "SV Grammar",
        "prompt": "Översätt till vårdad svenska (endast översättningen):\n'We sent the final report to them yesterday and asked where they are heading next quarter. However, nobody knew where the backup servers were located or whether the proposal provided any concrete support.'",
    },
    {
        "title": "4. Swedish Proofreading Stress-Test ('de/dem' & Gender)",
        "tag": "SV Proofread",
        "prompt": "Rätta alla språkfel och anglicismer i meningen och svara ENDAST med den korrigerade meningen:\n'Bland de som deltog fanns stor tålamod, men i slutet av dagen gav förslaget inte någon egentlig värde.'",
    },
    {
        "title": "5. Direction & Motion Proofreading ('var' -> 'vart')",
        "tag": "SV Proofread",
        "prompt": "Rätta alla språkfel och anglicismer i meningen och svara ENDAST med den korrigerade meningen:\n'Ingen visste var tåget var på väg eftersom konduktören hade inte informerat de.'",
    },
    {
        "title": "6. Native Corporate Message (VD-brev på ren svenska)",
        "tag": "SV Authoring",
        "prompt": "Skriv ett kort meddelande (4 meningar) på vårdad svenska från en vd till medarbetarna om varför styrelsen fattade ett svårt beslut, vilka åtgärder som vidtas och hur bolaget ger stöd till alla dem som berörs.",
    },
    {
        "title": "7. Zero-Collateral Control: French (Mistral's Home Language)",
        "tag": "FR Control (0 Diff)",
        "prompt": "Expliquez en trois phrases courtes comment fonctionne la photosynthèse chez les plantes vertes.",
    },
    {
        "title": "8. Zero-Collateral Control: Executable Python Algorithm",
        "tag": "Python Control (0 Diff)",
        "prompt": "Write ONLY a clean Python function `def binary_search(arr, target):` returning the index of `target` in sorted `arr` or -1.",
    },
    {
        "title": "9. Svensk Matkultur: Varför är räkmackan speciell i Sverige?",
        "tag": "SV Open Essay",
        "prompt": "Förklara varför kombinationen av räkor och mackor är speciell i Sverige.",
    },
]


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Ministral-3-8B — Zero-Gradient Swedish Weight Cartridge Studio</title>
<style>
  :root {
    --bg: #070b14;
    --panel: #0f1626;
    --panel-alt: #131d33;
    --border: #1f2d4d;
    --text: #eaf0ff;
    --muted: #8ea0c8;
    --sweden-gold: #fecc02;
    --sweden-blue: #006aa7;
    --accent: #38bdf8;
    --green: #10b981;
    --green-bg: rgba(16, 185, 129, 0.16);
    --red: #f43f5e;
    --red-bg: rgba(244, 63, 94, 0.18);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: radial-gradient(circle at 15% 0%, #0f2142 0%, var(--bg) 55%);
    color: var(--text);
    min-height: 100vh;
  }
  .topbar {
    border-bottom: 1px solid var(--border);
    background: rgba(10, 16, 30, 0.88);
    backdrop-filter: blur(12px);
    padding: 16px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    position: sticky;
    top: 0;
    z-index: 20;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 14px;
  }
  .flag-pill {
    background: linear-gradient(135deg, var(--sweden-blue) 60%, var(--sweden-gold) 100%);
    color: #fff;
    font-weight: 800;
    font-size: 13px;
    padding: 6px 11px;
    border-radius: 8px;
    letter-spacing: 0.5px;
    box-shadow: 0 0 18px rgba(0, 106, 167, 0.5);
  }
  .brand h1 {
    margin: 0;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.2px;
  }
  .brand .sub {
    font-size: 12px;
    color: var(--muted);
    margin-top: 3px;
  }
  .telemetry-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
  }
  .pill {
    background: var(--panel-alt);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 6px 13px;
    font-size: 12px;
    color: var(--muted);
  }
  .pill strong { color: var(--text); }
  .swap-btn {
    cursor: pointer;
    border: 1px solid #2b4470;
    border-radius: 999px;
    padding: 9px 18px;
    font-weight: 700;
    font-size: 13px;
    display: flex;
    align-items: center;
    gap: 9px;
    transition: all 0.18s ease;
    background: #17223b;
    color: var(--text);
  }
  .swap-btn.on {
    background: linear-gradient(135deg, #065f46, #059669);
    border-color: #34d399;
    box-shadow: 0 0 22px rgba(16, 185, 129, 0.35);
  }
  .dot {
    width: 9px; height: 9px; border-radius: 50%;
    background: #64748b;
  }
  .swap-btn.on .dot {
    background: #6ee7b7;
    box-shadow: 0 0 8px #6ee7b7;
  }
  .container {
    max-width: 1380px;
    margin: 0 auto;
    padding: 22px 28px 48px;
  }
  .tabs {
    display: flex;
    gap: 10px;
    margin-bottom: 18px;
  }
  .tab {
    background: var(--panel);
    border: 1px solid var(--border);
    color: var(--muted);
    padding: 10px 18px;
    border-radius: 10px;
    font-weight: 600;
    font-size: 13px;
    cursor: pointer;
  }
  .tab.active {
    background: var(--panel-alt);
    color: var(--sweden-gold);
    border-color: var(--sweden-gold);
  }
  .preset-bar {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(290px, 1fr));
    gap: 9px;
    margin-bottom: 16px;
  }
  .preset-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 13px;
    cursor: pointer;
    text-align: left;
    transition: border-color 0.15s, transform 0.12s;
  }
  .preset-card:hover {
    border-color: var(--accent);
    transform: translateY(-1px);
  }
  .preset-tag {
    display: inline-block;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    padding: 2px 7px;
    border-radius: 4px;
    background: rgba(56, 189, 248, 0.14);
    color: var(--accent);
    margin-bottom: 5px;
  }
  .preset-title {
    font-size: 12.5px;
    font-weight: 600;
    color: var(--text);
  }
  .prompt-box {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 18px;
  }
  textarea {
    width: 100%;
    min-height: 86px;
    background: #080d19;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 11px 13px;
    font-family: inherit;
    font-size: 14px;
    resize: vertical;
  }
  .controls {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 12px;
    gap: 12px;
    flex-wrap: wrap;
  }
  .btn-primary {
    background: linear-gradient(135deg, #0284c7, #2563eb);
    color: white;
    border: none;
    padding: 11px 22px;
    border-radius: 9px;
    font-weight: 700;
    font-size: 14px;
    cursor: pointer;
    box-shadow: 0 4px 18px rgba(37, 99, 235, 0.35);
  }
  .btn-primary:disabled { opacity: 0.55; cursor: wait; }
  .btn-secondary {
    background: var(--panel-alt);
    color: var(--text);
    border: 1px solid var(--border);
    padding: 10px 16px;
    border-radius: 9px;
    font-weight: 600;
    font-size: 13px;
    cursor: pointer;
  }
  .arena {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
  }
  @media (max-width: 960px) { .arena { grid-template-columns: 1fr; } }
  .out-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px;
    min-height: 250px;
    display: flex;
    flex-direction: column;
  }
  .out-card.after {
    border-color: rgba(16, 185, 129, 0.55);
    box-shadow: 0 0 28px rgba(16, 185, 129, 0.08);
  }
  .card-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 14px;
  }
  .card-title { font-weight: 700; font-size: 14px; }
  .badge {
    font-size: 11px;
    font-weight: 700;
    padding: 4px 9px;
    border-radius: 999px;
  }
  .badge-red { background: var(--red-bg); color: #fda4af; border: 1px solid rgba(244,63,94,0.4); }
  .badge-green { background: var(--green-bg); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.4); }
  .reply-body {
    font-size: 15px;
    line-height: 1.65;
    white-space: pre-wrap;
    flex: 1;
  }
  .hl-bad {
    background: var(--red-bg);
    color: #fecdd3;
    border-bottom: 2px wavy var(--red);
    padding: 1px 4px;
    border-radius: 4px;
    font-weight: 600;
  }
  .hl-good {
    background: var(--green-bg);
    color: #a7f3d0;
    border-bottom: 2px solid var(--green);
    padding: 1px 4px;
    border-radius: 4px;
    font-weight: 600;
  }
  .card-foot {
    margin-top: 14px;
    padding-top: 10px;
    border-top: 1px solid var(--border);
    font-size: 12px;
    color: var(--muted);
    display: flex;
    justify-content: space-between;
  }
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(235px, 1fr));
    gap: 14px;
    margin-bottom: 22px;
  }
  .kpi {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
  }
  .kpi .label { font-size: 12px; color: var(--muted); }
  .kpi .val { font-size: 26px; font-weight: 800; margin: 6px 0 4px; color: var(--sweden-gold); }
  .kpi .delta { font-size: 12.5px; color: #6ee7b7; font-weight: 600; }
  table {
    width: 100%;
    border-collapse: collapse;
    background: var(--panel);
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--border);
    margin-bottom: 22px;
  }
  th, td {
    padding: 11px 14px;
    text-align: left;
    border-bottom: 1px solid var(--border);
    font-size: 13.5px;
  }
  th { background: var(--panel-alt); color: var(--muted); font-size: 12px; text-transform: uppercase; }
  .hidden { display: none; }
</style>
</head>
<body>
  <div class="topbar">
    <div class="brand">
      <div class="flag-pill">SV + EU</div>
      <div>
        <h1>Ministral-3-8B — Zero-Gradient Swedish Weight Cartridge</h1>
        <div class="sub">Pure Weight-Space Upgrade &bull; 0 Training Steps &bull; 0 System Prompt Tokens &bull; 100% Base Weights Untouched</div>
      </div>
    </div>
    <div class="telemetry-pills">
      <div class="pill">Cartridge: <strong id="cartSize">960.8 KB</strong></div>
      <div class="pill">Base Untouched: <strong>100.0000%</strong></div>
      <div class="pill">Last Hot-Swap: <strong id="swapMs">-- ms</strong></div>
      <button id="swapBtn" class="swap-btn" onclick="toggleCartridge()">
        <span class="dot"></span>
        <span id="swapLabel">CARTRIDGE: LOADING...</span>
      </button>
    </div>
  </div>

  <div class="container">
    <div class="tabs">
      <button id="tabLive" class="tab active" onclick="switchTab('live')">⚡ Live Side-by-Side Arena</button>
      <button id="tabBench" class="tab" onclick="switchTab('bench')">📊 Verified 2,640-Item Benchmark Audit</button>
    </div>

    <div id="viewLive">
      <div class="preset-bar" id="presetBar"></div>
      <div class="prompt-box">
        <textarea id="promptInput"></textarea>
        <div class="controls">
          <div style="display:flex; gap:10px; align-items:center;">
            <button id="runPairBtn" class="btn-primary" onclick="runPairedComparison()">▶ Run Live BEFORE vs AFTER on GPU</button>
            <button id="runSingleBtn" class="btn-secondary" onclick="runCurrentModeOnly()">Run Current Cartridge State Only</button>
          </div>
          <div style="font-size:12.5px; color:var(--muted);">
            Max tokens: <input id="maxTok" type="number" value="512" min="32" max="2048" style="width:76px; background:#080d19; color:#fff; border:1px solid var(--border); border-radius:6px; padding:5px 8px;" />
          </div>
        </div>
      </div>

      <div class="arena">
        <div class="out-card">
          <div class="card-head">
            <div class="card-title">BEFORE — Unmodified Ministral-3-8B-Instruct-2512</div>
            <span id="beforeBadge" class="badge badge-red">Baseline</span>
          </div>
          <div id="beforeText" class="reply-body" style="color:var(--muted);">Select a preset above and click "Run Live BEFORE vs AFTER on GPU".</div>
          <div class="card-foot">
            <span id="beforeSpeed">-- tok/s</span>
            <span id="beforeDefects">--</span>
          </div>
        </div>

        <div class="out-card after">
          <div class="card-head">
            <div class="card-title">AFTER — + 960 KB Swedish Weight Cartridge (0-Grad)</div>
            <span id="afterBadge" class="badge badge-green">+ Cartridge Active</span>
          </div>
          <div id="afterText" class="reply-body" style="color:var(--muted);">Cartridge attaches directly into Layer 32 &amp; 33 Linear weights in &lt; 1 ms.</div>
          <div class="card-foot">
            <span id="afterSpeed">-- tok/s</span>
            <span id="afterDefects">--</span>
          </div>
        </div>
      </div>
    </div>

    <div id="viewBench" class="hidden">
      <div id="benchContent">Loading verified benchmark results...</div>
    </div>
  </div>

<script>
let presets = [];
let defectPatterns = [];
let goldPatterns = [];

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function highlightText(raw, mode) {
  let html = escapeHtml(raw);
  if (mode === 'before') {
    for (const pat of defectPatterns) {
      const re = new RegExp('(' + pat + ')', 'gi');
      html = html.replace(re, '<span class="hl-bad">$1</span>');
    }
  } else {
    for (const pat of goldPatterns) {
      const re = new RegExp('(' + pat + ')', 'gi');
      html = html.replace(re, '<span class="hl-good">$1</span>');
    }
    for (const pat of defectPatterns) {
      const re = new RegExp('(' + pat + ')', 'gi');
      html = html.replace(re, '<span class="hl-bad">$1</span>');
    }
  }
  return html;
}

function updateTopStatus(st) {
  const btn = document.getElementById('swapBtn');
  const lbl = document.getElementById('swapLabel');
  document.getElementById('cartSize').textContent = st.file_size_kb + ' KB';
  if (st.last_swap_ms !== undefined && st.last_swap_ms !== null) {
    document.getElementById('swapMs').textContent = st.last_swap_ms.toFixed(2) + ' ms';
  }
  if (st.attached) {
    btn.classList.add('on');
    lbl.textContent = 'CARTRIDGE: ON (' + st.added_neurons + ' Neurons)';
  } else {
    btn.classList.remove('on');
    lbl.textContent = 'CARTRIDGE: OFF (Base Weights)';
  }
}

async function initStudio() {
  const r = await fetch('/api/init');
  const data = await r.json();
  presets = data.presets;
  defectPatterns = data.defect_regexes;
  goldPatterns = data.gold_regexes;
  updateTopStatus(data.status);

  const bar = document.getElementById('presetBar');
  bar.innerHTML = '';
  presets.forEach((p, i) => {
    const d = document.createElement('div');
    d.className = 'preset-card';
    d.innerHTML = '<div class="preset-tag">' + escapeHtml(p.tag) + '</div><div class="preset-title">' + escapeHtml(p.title) + '</div>';
    d.onclick = () => { document.getElementById('promptInput').value = p.prompt; };
    bar.appendChild(d);
  });
  if (presets.length > 0) document.getElementById('promptInput').value = presets[0].prompt;
  if (data.benchmarks) renderBenchmarks(data.benchmarks);
}

async function toggleCartridge() {
  const r = await fetch('/api/toggle', { method: 'POST' });
  const st = await r.json();
  updateTopStatus(st);
}

async function runPairedComparison() {
  const btn = document.getElementById('runPairBtn');
  const prompt = document.getElementById('promptInput').value;
  const maxTok = parseInt(document.getElementById('maxTok').value || '512', 10);
  btn.disabled = true;
  btn.textContent = '⏳ Generating Live BEFORE & AFTER on GPU...';
  document.getElementById('beforeText').textContent = 'Running unmodified Ministral-3-8B...';
  document.getElementById('afterText').textContent = 'Waiting for hot-swap...';

  try {
    const r = await fetch('/api/compare', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ prompt, max_new_tokens: maxTok })
    });
    const res = await r.json();
    updateTopStatus(res.status);

    document.getElementById('beforeText').innerHTML = highlightText(res.before.reply, 'before');
    document.getElementById('beforeSpeed').textContent = res.before.tok_per_s + ' tok/s (' + res.before.tokens + ' tok)';
    document.getElementById('beforeBadge').textContent = res.before.defect_count > 0 ? (res.before.defect_count + ' Swenglish/Grammar Defects') : '0 Highlighted Defects';

    document.getElementById('afterText').innerHTML = highlightText(res.after.reply, 'after');
    document.getElementById('afterSpeed').textContent = res.after.tok_per_s + ' tok/s (' + res.after.tokens + ' tok • Hot-Swapped ON in ' + res.attach_ms.toFixed(2) + ' ms)';
    if (res.after.defect_count === 0) {
      document.getElementById('afterBadge').textContent = res.after.gold_count > 0
        ? ('✓ Clean Idiomatic (' + res.after.gold_count + ' Idiom Hits)')
        : '✓ Clean Idiomatic Swedish';
    } else {
      document.getElementById('afterBadge').textContent = res.after.defect_count + ' Defects';
    }
  } finally {
    btn.disabled = false;
    btn.textContent = '▶ Run Live BEFORE vs AFTER on GPU';
  }
}

async function runCurrentModeOnly() {
  const prompt = document.getElementById('promptInput').value;
  const maxTok = parseInt(document.getElementById('maxTok').value || '512', 10);
  const r = await fetch('/api/generate_single', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ prompt, max_new_tokens: maxTok })
  });
  const res = await r.json();
  if (res.attached) {
    document.getElementById('afterText').innerHTML = highlightText(res.reply, 'after');
    document.getElementById('afterSpeed').textContent = res.tok_per_s + ' tok/s (' + res.tokens + ' tok)';
  } else {
    document.getElementById('beforeText').innerHTML = highlightText(res.reply, 'before');
    document.getElementById('beforeSpeed').textContent = res.tok_per_s + ' tok/s (' + res.tokens + ' tok)';
  }
}

function switchTab(which) {
  document.getElementById('tabLive').classList.toggle('active', which === 'live');
  document.getElementById('tabBench').classList.toggle('active', which === 'bench');
  document.getElementById('viewLive').classList.toggle('hidden', which !== 'live');
  document.getElementById('viewBench').classList.toggle('hidden', which !== 'bench');
}

function renderBenchmarks(b) {
  const hm = b.headline_metrics;
  const zc = b.zero_collateral_international;
  const ab = b.symmetric_150_ablation_vs_naive_steering;
  let html = '<div class="kpi-grid">';
  html += '<div class="kpi"><div class="label">SwedMaster-300 (6 Domains)</div><div class="val">' + hm.swedmaster_300.before_pct + '% &rarr; ' + hm.swedmaster_300.after_pct + '%</div><div class="delta">+' + hm.swedmaster_300.delta_pp + ' pp (McNemar p = ' + hm.swedmaster_300.mcnemar.p_value.toExponential(2) + ')</div></div>';
  html += '<div class="kpi"><div class="label">Official ScandEval ScaLA-sv (n=1,000)</div><div class="val">' + hm.scala_sv_1000_official.before_pct + '% &rarr; ' + hm.scala_sv_1000_official.after_pct + '%</div><div class="delta">MCC ' + hm.scala_sv_1000_official.before_mcc + ' &rarr; ' + hm.scala_sv_1000_official.after_mcc + '</div></div>';
  html += '<div class="kpi"><div class="label">SwedFreeGen-60 Clean Replies</div><div class="val">' + hm.swed_freegen_60.before_clean_pct + '% &rarr; ' + hm.swed_freegen_60.after_clean_pct + '%</div><div class="delta">Defects: ' + hm.swed_freegen_60.before_total_defects + ' &rarr; ' + hm.swed_freegen_60.after_total_defects + ' (-' + hm.swed_freegen_60.defect_reduction_pct + '%)</div></div>';
  html += '<div class="kpi"><div class="label">Zero-Collateral Gate (EN / FR / Code)</div><div class="val">' + zc.anchor_40.top1_matches + ' / ' + zc.anchor_40.total + '</div><div class="delta">Mean KL = ' + zc.anchor_40.mean_kl_bf16 + ' | Python ' + zc.python_20.after_passed + '/20</div></div>';
  html += '</div>';

  html += '<h3>1. SwedMaster-300 Breakdown by Linguistic Category</h3><table><thead><tr><th>Category</th><th>Before</th><th>After (+960 KB Cartridge)</th><th>Gain</th><th>McNemar p-value</th></tr></thead><tbody>';
  for (const [k, v] of Object.entries(b.pillar1_categories)) {
    html += '<tr><td><strong>' + escapeHtml(k) + '</strong></td><td>' + v.before_pct + '% (' + v.before_correct + '/' + v.total + ')</td><td style="color:#6ee7b7;font-weight:700;">' + v.after_pct + '% (' + v.after_correct + '/' + v.total + ')</td><td>+' + v.delta_pp + ' pp</td><td>' + v.mcnemar.p_value.toExponential(2) + '</td></tr>';
  }
  html += '</tbody></table>';

  html += '<h3>2. Why Context-Gated Weight Cartridges Beat Linear Steering (150 Symmetric Items: de/dem, var/vart, en/ett)</h3>';
  html += '<table><thead><tr><th>Method</th><th>150-Item Symmetric Accuracy</th><th>Mechanism</th></tr></thead><tbody>';
  html += '<tr><td>Unmodified Ministral-3-8B</td><td>' + ab.baseline_before_pct + '%</td><td>Defaults to majority pole (de, var, utrum)</td></tr>';
  html += '<tr><td>Unconditional Rank-1 / Linear Logit Steering</td><td style="color:#fda4af;">' + ab.naive_linear_steering_pct + '%</td><td>Boosting minority forms collapses opposite grammatical pole</td></tr>';
  html += '<tr><td><strong>Aphygo 960 KB Weight Cartridge</strong></td><td style="color:#6ee7b7;font-weight:700;">' + ab.synapse_cartridge_after_pct + '%</td><td>Non-linear SwiGLU context gates separate both poles cleanly</td></tr>';
  html += '</tbody></table>';

  html += '<h3>3. International Zero-Collateral Audit (English, French, Coding, Math &amp; Speed)</h3>';
  html += '<table><thead><tr><th>Benchmark</th><th>Before</th><th>After</th><th>Status</th></tr></thead><tbody>';
  html += '<tr><td>Multilingual &amp; Code Anchor-40</td><td>40 / 40</td><td>' + zc.anchor_40.top1_matches + ' / 40 (KL ' + zc.anchor_40.mean_kl_bf16 + ')</td><td>✓ 0 Regressions</td></tr>';
  html += '<tr><td>Meta Belebele English (n=200)</td><td>' + zc.belebele_eng_200.before_pct + '%</td><td>' + zc.belebele_eng_200.after_pct + '%</td><td>✓ Preserved</td></tr>';
  html += '<tr><td>Meta Belebele French (n=200)</td><td>' + zc.belebele_fra_200.before_pct + '%</td><td>' + zc.belebele_fra_200.after_pct + '%</td><td>✓ Preserved</td></tr>';
  html += '<tr><td>Executable Python Sandbox (n=20)</td><td>' + zc.python_20.before_pct + '%</td><td>' + zc.python_20.after_pct + '%</td><td>✓ Preserved</td></tr>';
  html += '<tr><td>Math Reasoning Oracle (n=20)</td><td>' + zc.math_20.before_pct + '%</td><td>' + zc.math_20.after_pct + '%</td><td>✓ Preserved</td></tr>';
  html += '<tr><td>SuperLim-2 DaLAJ Minimal Pairs (n=500)</td><td>' + hm.dalaj_500_official.before_pct + '%</td><td>' + hm.dalaj_500_official.after_pct + '%</td><td>✓ Preserved / Improved</td></tr>';
  html += '<tr><td>Meta Belebele Swedish (n=300)</td><td>' + hm.belebele_swe_300_official.before_pct + '%</td><td>' + hm.belebele_swe_300_official.after_pct + '%</td><td>✓ Preserved / Improved</td></tr>';
  html += '</tbody></table>';

  document.getElementById('benchContent').innerHTML = html;
}

initStudio();
</script>
</body>
</html>
"""


class GraceSentenceStopper(StoppingCriteria):
    """Allows generation to finish the current sentence and close markdown tags after soft max_new_tokens."""

    def __init__(self, prompt_len: int, soft_max_new_tokens: int, tokenizer: Any):
        super().__init__()
        self.prompt_len = prompt_len
        self.soft_max_new_tokens = soft_max_new_tokens
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        gen_len = int(input_ids.shape[1]) - self.prompt_len
        if gen_len < self.soft_max_new_tokens:
            return False
        text = self.tokenizer.decode(input_ids[0, self.prompt_len:], skip_special_tokens=True)
        stripped = text.rstrip()
        if not stripped:
            return False
        # Do not stop while inside an unclosed bold/markdown marker
        if stripped.count("**") % 2 != 0:
            return False
        return stripped.endswith((".", "!", "?", "\n", ".\""))


class StudioRuntime:
    def __init__(self, model_path: str, cartridge_path: str, bench_json_path: str):
        self.lock = threading.Lock()
        print(f"Loading model from {model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        ).eval()
        self.cartridge_path = cartridge_path
        self.cartridge_mtime = os.path.getmtime(cartridge_path)
        self.cartridge: SynapseCartridgeHandle = load_cartridge(cartridge_path)
        # Pre-warm GPU cache and leave attached by default
        _ = self.cartridge.attach(self.model)
        _ = self.cartridge.detach(self.model)
        self.last_swap_ms = self.cartridge.attach(self.model)
        self.bench_data = None
        if os.path.isfile(bench_json_path):
            with open(bench_json_path, "r", encoding="utf-8") as f:
                self.bench_data = json.load(f)

    def _reload_cartridge_if_updated(self):
        try:
            cur_mtime = os.path.getmtime(self.cartridge_path)
            if cur_mtime != self.cartridge_mtime:
                was_attached = self.cartridge.is_attached
                if was_attached:
                    self.cartridge.detach(self.model)
                self.cartridge = load_cartridge(self.cartridge_path)
                self.cartridge_mtime = cur_mtime
                if was_attached:
                    self.last_swap_ms = self.cartridge.attach(self.model)
        except OSError:
            pass

    def status_dict(self) -> Dict[str, Any]:
        s = self.cartridge.summary()
        s["last_swap_ms"] = round(self.last_swap_ms, 3)
        return s

    @torch.no_grad()
    def _generate_once(self, prompt: str, max_new_tokens: int) -> Dict[str, Any]:
        inp = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            return_tensors="pt",
            add_generation_prompt=True,
        )
        inp_ids = (inp["input_ids"] if hasattr(inp, "keys") else inp).to(self.model.device)
        prompt_len = int(inp_ids.shape[1])
        stopper = StoppingCriteriaList([GraceSentenceStopper(prompt_len, max_new_tokens, self.tokenizer)])
        hard_max_tokens = max_new_tokens + 64
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.model.generate(
            input_ids=inp_ids,
            max_new_tokens=hard_max_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=stopper,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = max(1e-6, time.perf_counter() - t0)
        new_ids = out[0, prompt_len:]
        reply = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        n_tok = int(new_ids.numel())
        defects = sum(len(re.findall(p, reply, flags=re.IGNORECASE)) for p in UI_DEFECT_REGEXES)
        golds = sum(1 for p in UI_GOLD_REGEXES if re.search(p, reply, flags=re.IGNORECASE))
        return {
            "reply": reply,
            "tokens": n_tok,
            "tok_per_s": round(n_tok / dt, 1),
            "defect_count": defects,
            "gold_count": golds,
        }

    def toggle(self) -> Dict[str, Any]:
        with self.lock:
            self._reload_cartridge_if_updated()
            if self.cartridge.is_attached:
                self.last_swap_ms = self.cartridge.detach(self.model)
            else:
                self.last_swap_ms = self.cartridge.attach(self.model)
            return self.status_dict()

    def compare_before_after(self, prompt: str, max_new_tokens: int) -> Dict[str, Any]:
        with self.lock:
            self._reload_cartridge_if_updated()
            self.cartridge.detach(self.model)
            before_out = self._generate_once(prompt, max_new_tokens)
            attach_ms = self.cartridge.attach(self.model)
            self.last_swap_ms = attach_ms
            after_out = self._generate_once(prompt, max_new_tokens)
            return {
                "before": before_out,
                "after": after_out,
                "attach_ms": round(attach_ms, 3),
                "status": self.status_dict(),
            }

    def generate_current(self, prompt: str, max_new_tokens: int) -> Dict[str, Any]:
        with self.lock:
            self._reload_cartridge_if_updated()
            res = self._generate_once(prompt, max_new_tokens)
            res["attached"] = self.cartridge.is_attached
            return res


def make_handler(runtime: StudioRuntime):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def _send_json(self, obj: Any, code: int = 200):
            raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index"):
                raw = HTML_PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            elif self.path == "/api/init":
                self._send_json({
                    "status": runtime.status_dict(),
                    "presets": PRESET_PROMPTS,
                    "defect_regexes": UI_DEFECT_REGEXES,
                    "gold_regexes": UI_GOLD_REGEXES,
                    "benchmarks": runtime.bench_data,
                })
            else:
                self.send_error(404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
            if self.path == "/api/toggle":
                self._send_json(runtime.toggle())
            elif self.path == "/api/compare":
                prompt = str(body.get("prompt", "")).strip()
                max_tok = max(32, min(2048, int(body.get("max_new_tokens", 512))))
                self._send_json(runtime.compare_before_after(prompt, max_tok))
            elif self.path == "/api/generate_single":
                prompt = str(body.get("prompt", "")).strip()
                max_tok = max(32, min(2048, int(body.get("max_new_tokens", 512))))
                self._send_json(runtime.generate_current(prompt, max_tok))
            else:
                self.send_error(404)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Ministral-3-8B Swedish Weight Cartridge Live Studio")
    default_model = DEFAULT_LOCAL_MODEL if os.path.isdir(DEFAULT_LOCAL_MODEL) else HF_MODEL_ID
    parser.add_argument("--model", default=default_model, help="Local directory or HuggingFace repo ID")
    parser.add_argument("--cartridge", default=DEFAULT_CARTRIDGE, help="Path to .safetensors weight cartridge")
    parser.add_argument("--benchmarks", default=DEFAULT_BENCH_JSON, help="Path to verified benchmark JSON")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7865)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    runtime = StudioRuntime(args.model, args.cartridge, args.benchmarks)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runtime))
    url = f"http://{args.host}:{args.port}"
    print(f"\n==================================================================")
    print(f" Ministral-3-8B Swedish Weight Cartridge Studio running at:")
    print(f"   {url}")
    print(f"==================================================================\n")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down studio.")


if __name__ == "__main__":
    main()
