"""Operational Streamlit application for radiology-summary integrity auditing."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import sys
import threading
import time
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
SOURCE_ENTRY = str(SOURCE_ROOT)
while SOURCE_ENTRY in sys.path:
    sys.path.remove(SOURCE_ENTRY)
sys.path.insert(0, SOURCE_ENTRY)

# A long-lived Streamlit Cloud worker can retain modules from a previously
# installed project package. Remove them before importing so the checksum
# validator and inference engine always come from this sealed repository.
for module_name in tuple(sys.modules):
    if module_name == "radtrace" or module_name.startswith("radtrace."):
        sys.modules.pop(module_name, None)

from radtrace import release as release_runtime
from radtrace.bidirectional import (
    BidirectionalAuditor,
    BidirectionalAuditResult,
    DirectionRisk,
    ReportRiskCalibrator,
    ReportRiskFeatures,
)
from radtrace.privacy import scan_privacy_risk
from radtrace.product import (
    ENV_ARTIFACT,
    ENV_CALIBRATOR,
    ENV_DEVICE,
    ENV_EVALUATION_LEDGER,
    ENV_RELEASE_MANIFEST,
    EXPORT_PREFIX,
    PRODUCT_EXPANSION,
    PRODUCT_NAME,
    RESEARCH_NOTICE,
)
from radtrace.showcase import SHOWCASE_CASES, SHOWCASE_CASES_BY_ID, showcase_dataset_csv
from radtrace.types import AuditResult
from radtrace.ui_components import (
    LABEL_META,
    claim_card_html,
    evidence_panel_html,
    normalize_audit_result,
    result_json,
    summarize_claims,
)
from radtrace.upload import (
    MAX_UPLOAD_CASES,
    UploadValidationError,
    parse_uploaded_cases,
    upload_template_csv,
)
from radtrace.verifier import RadTraceVerifier


def _runtime_path(variable: str, default: str) -> Path:
    configured = Path(os.getenv(variable, default)).expanduser()
    return configured if configured.is_absolute() else PROJECT_ROOT / configured


DEFAULT_ARTIFACT = _runtime_path(ENV_ARTIFACT, "artifacts/mirror-rad-system/tuned_verifier")
DEFAULT_CALIBRATOR = _runtime_path(ENV_CALIBRATOR, "artifacts/mirror-rad-risk-calibrator.pkl")
EVALUATION_SUMMARY = _runtime_path(
    ENV_EVALUATION_LEDGER,
    "reports/public/evaluation_summary.json",
)
RELEASE_MANIFEST = _runtime_path(ENV_RELEASE_MANIFEST, "artifacts/mirror-rad-release.json")
MAX_TEXT_CHARACTERS = 40_000
BATCH_REPORTS = max(1, int(os.getenv("MIRROR_RAD_BATCH_REPORTS", "8")))
NLI_BATCH_SIZE = max(1, int(os.getenv("MIRROR_RAD_NLI_BATCH_SIZE", "8")))
AUDIT_SCHEMA = "mirror-rad.bidirectional-audit.v1"
BATCH_SCHEMA = "mirror-rad.batch-audit.v1"
RELEASE_SCHEMA = "mirror-rad.release.v1"
INFERENCE_GATE = threading.BoundedSemaphore(value=1)

st.set_page_config(
    page_title=f"{PRODUCT_NAME} | Radiology integrity audit",
    page_icon="↔",
    layout="wide",
    initial_sidebar_state="collapsed",
)


CSS = r"""
<style>
:root {
  --mr-canvas:#f5f6fa;--mr-surface:#ffffff;--mr-soft:#f0f2f8;--mr-obsidian:#080b18;
  --mr-midnight:#111a32;--mr-ink:#171b2e;--mr-body:#3c445d;--mr-muted:#5f677e;
  --mr-line:#dfe2eb;--mr-line-strong:#c6cada;--mr-iris:#6558f5;--mr-iris-dark:#4939b8;
  --mr-iris-soft:#eeecff;--mr-cyan:#11b8d2;--mr-cyan-soft:#e3f8fb;--mr-mint:#159c86;
  --mr-good:#137052;--mr-good-soft:#e8f7f1;--mr-warn:#7a4b00;--mr-warn-soft:#fff2d6;
  --mr-bad:#9d2828;--mr-bad-soft:#ffeded;--mr-focus:#5245d7;--mr-radius:20px;
  --mr-shadow:0 18px 55px rgba(24,29,61,.09),0 2px 8px rgba(24,29,61,.04);
  --mr-font:"Segoe UI Variable Text","Segoe UI Variable","Segoe UI",Inter,Roboto,"Helvetica Neue",Arial,sans-serif;
  --mr-display:"Segoe UI Variable Display","Segoe UI Variable","Segoe UI",Inter,Roboto,"Helvetica Neue",Arial,sans-serif;
}
html{font-size:16px;scroll-behavior:smooth}html,body,[class*="css"],.stApp{font-family:var(--mr-font);font-synthesis:none;-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
.stApp p,.stApp li,.stApp label,.stApp button,.stApp input,.stApp textarea,.stApp select,.stApp [role="tab"],.stApp [data-testid="stMarkdownContainer"],.mirror-brandbar,.st-key-mirror_navigation,.app-head,.notice,.section-head,.step-grid,.input-panel-title,.risk-hero,.kpi-grid,.dataset-card,.research-card,.benchmark-card,.claim-card,.readiness-hero{font-family:var(--mr-font)!important}
.stApp h1,.stApp h2,.stApp h3,.mirror-brandcopy strong,.section-title,.risk-hero h2,.readiness-hero h2{font-family:var(--mr-display)!important}
[data-testid="stIconMaterial"],.material-symbols-rounded,.material-symbols-outlined{font-family:"Material Symbols Rounded","Material Symbols Outlined"!important}
body,.stApp{color:var(--mr-body)}.stApp,[data-testid="stAppViewContainer"]{background:var(--mr-canvas)}
[data-testid="stHeader"]{height:2.9rem;background:rgba(245,246,250,.8);backdrop-filter:blur(18px);border-bottom:1px solid rgba(223,226,235,.7)}
[data-testid="stToolbar"]{color:var(--mr-muted)}[data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none!important}
.block-container{max-width:1420px;width:100%;padding:4rem 2.2rem 5.5rem;overflow-x:hidden}

/* MIRROR command shell */
.mirror-brandbar{position:relative;overflow:hidden;display:flex;justify-content:space-between;align-items:center;gap:1.4rem;min-height:82px;padding:.95rem 1.15rem;background:linear-gradient(135deg,rgba(255,255,255,.98),rgba(249,251,255,.97));border:1px solid var(--mr-line);border-radius:22px 22px 7px 7px;box-shadow:0 13px 38px rgba(25,31,67,.08);isolation:isolate}
.mirror-brandbar::before{content:"";position:absolute;right:-48px;top:-85px;width:250px;height:250px;border-radius:50%;background:radial-gradient(circle,rgba(101,88,245,.11),rgba(17,184,210,.035) 42%,transparent 68%);z-index:-1}
.mirror-brandbar::after{content:"";position:absolute;left:1.15rem;right:1.15rem;bottom:0;height:1px;background:linear-gradient(90deg,var(--mr-cyan),rgba(101,88,245,.42),transparent 72%);z-index:-1}
.mirror-brand{display:flex;align-items:center;gap:.9rem;min-width:0}.mirror-mark{position:relative;flex:0 0 auto;width:50px;height:50px;border:1px solid rgba(101,88,245,.28);border-radius:15px;background:linear-gradient(145deg,#111a35,#312a78);box-shadow:inset 0 1px 0 rgba(255,255,255,.14),0 9px 24px rgba(42,42,105,.18)}
.mirror-mark::before,.mirror-mark::after{content:"";position:absolute;top:12px;width:11px;height:27px;border-top:3px solid #fff;border-bottom:3px solid var(--mr-cyan)}
.mirror-mark::before{left:12px;border-left:3px solid #fff;transform:skewY(32deg)}.mirror-mark::after{right:12px;border-right:3px solid #fff;transform:skewY(-32deg)}
.mirror-mark span{position:absolute;left:50%;top:8px;bottom:8px;width:1px;background:linear-gradient(transparent,#79e6f4,transparent);box-shadow:0 0 10px #49d5e8}
.mirror-brandcopy strong{display:block;color:var(--mr-ink);font-family:var(--mr-display);font-size:1.22rem;line-height:1.15;font-weight:760;letter-spacing:-.025em}.mirror-brandcopy span{display:block;color:var(--mr-muted);font-size:.73rem;line-height:1.42;margin-top:.22rem;max-width:570px;overflow-wrap:anywhere}
.mirror-command-meta{display:flex;align-items:center;gap:.55rem;flex-wrap:wrap;justify-content:flex-end}.command-chip{display:inline-flex;align-items:center;gap:.48rem;min-height:34px;padding:.38rem .68rem;border:1px solid var(--mr-line);border-radius:999px;background:#fff;color:var(--mr-body);font-size:.73rem;line-height:1.3;font-weight:650;white-space:nowrap;box-shadow:0 4px 13px rgba(29,35,72,.045)}.command-chip b{color:var(--mr-ink);font-weight:700}.command-chip .health-dot{width:8px;height:8px;border-radius:50%;background:#19a77e;box-shadow:0 0 0 4px rgba(25,167,126,.1);animation:health-pulse 2.4s ease-in-out infinite}.command-chip.attention .health-dot{background:#d49427;box-shadow:0 0 0 4px rgba(212,148,39,.12);animation:none}
.st-key-mirror_navigation{box-sizing:border-box;width:100%;margin-top:.48rem;padding:.24rem .32rem;background:rgba(255,255,255,.97);border:1px solid var(--mr-line);border-radius:14px;box-shadow:0 8px 24px rgba(26,31,63,.055);backdrop-filter:blur(16px);overflow:hidden}
.st-key-mirror_navigation [data-testid="stElementContainer"],.st-key-mirror_navigation [data-testid="stRadio"]{box-sizing:border-box;width:100%!important;max-width:100%!important;min-width:0!important}
.st-key-mirror_navigation div[role="radiogroup"]{display:grid;width:100%;grid-template-columns:repeat(5,minmax(0,1fr));align-items:stretch;gap:.18rem;overflow:visible;padding:0}
.st-key-mirror_navigation div[role="radiogroup"] label{position:relative;box-sizing:border-box;display:flex!important;align-items:center!important;justify-content:center!important;min-width:0;min-height:44px;padding:.57rem .62rem;border:0;border-radius:9px;margin:0;text-align:center;transition:background .18s ease,color .18s ease,box-shadow .18s ease}
.st-key-mirror_navigation div[role="radiogroup"] label:hover{background:#f5f6fa}.st-key-mirror_navigation div[role="radiogroup"] label:has(input:checked){background:linear-gradient(135deg,rgba(93,80,230,.07),rgba(14,169,192,.045));box-shadow:none}.st-key-mirror_navigation div[role="radiogroup"] label:has(input:checked)::after{content:"";position:absolute;left:50%;bottom:3px;width:30px;height:2.5px;border-radius:999px;transform:translateX(-50%);background:linear-gradient(90deg,var(--mr-iris),var(--mr-cyan));box-shadow:0 2px 8px rgba(93,80,230,.18)}.st-key-mirror_navigation div[role="radiogroup"] label:has(input:focus-visible){outline:none;box-shadow:inset 0 0 0 2px var(--mr-focus)}
.st-key-mirror_navigation div[role="radiogroup"] input{position:absolute;opacity:0;pointer-events:none}
.st-key-mirror_navigation label[data-testid="stRadioOption"]>div>div>div:first-child{display:none}
.st-key-mirror_navigation label[data-testid="stRadioOption"] [data-testid="stMarkdownContainer"]{display:flex!important;align-items:center!important;justify-content:center!important;width:100%!important;min-width:0!important;text-align:center!important}
.st-key-mirror_navigation div[role="radiogroup"] p{width:100%;max-width:100%;margin:0!important;color:#4d556c!important;font-size:.82rem;font-weight:690;line-height:1.25;text-align:center!important;white-space:nowrap}.st-key-mirror_navigation div[role="radiogroup"] label:has(input:checked) p{color:#312675!important;font-weight:760}

/* Hero: two evidence directions, one decision */
.app-head{position:relative;isolation:isolate;display:grid;grid-template-columns:minmax(0,1.15fr) minmax(360px,.85fr);gap:2.4rem;align-items:center;overflow:hidden;margin:1.15rem 0 1rem;padding:2rem 2.15rem;min-height:260px;background:radial-gradient(circle at 84% 10%,rgba(101,88,245,.3),transparent 28%),radial-gradient(circle at 54% 105%,rgba(17,184,210,.16),transparent 35%),linear-gradient(132deg,#0a0d1b 5%,#10172e 58%,#141137);border:1px solid rgba(110,115,180,.25);border-radius:28px;box-shadow:0 24px 65px rgba(20,24,56,.17)}
.app-head::before{content:"";position:absolute;inset:0;background-image:linear-gradient(rgba(255,255,255,.025) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.025) 1px,transparent 1px);background-size:32px 32px;mask-image:linear-gradient(90deg,#000,transparent 72%);z-index:-1}.app-head::after{content:"";position:absolute;right:32%;top:-35%;width:90px;height:170%;transform:rotate(20deg);background:linear-gradient(90deg,transparent,rgba(80,220,239,.08),transparent);animation:mirror-sweep 10s ease-in-out infinite;z-index:-1}
.hero-copy{min-width:0}.page-kicker{display:inline-flex;align-items:center;gap:.45rem;margin-bottom:.65rem;color:#81e5f1;font-size:.72rem;line-height:1.35;letter-spacing:.13em;text-transform:uppercase;font-weight:750}.page-kicker::before{content:"";width:22px;height:1px;background:currentColor;box-shadow:0 0 8px currentColor}.app-head h1{font-family:var(--mr-display);font-size:clamp(2.05rem,3.4vw,3.55rem);letter-spacing:-.055em;color:#fff;font-weight:760;margin:0;line-height:1.02;max-width:780px}.app-head p{margin:.78rem 0 0;color:#b8c1dc;max-width:720px;font-size:1rem;line-height:1.58}.app-head h1,.app-head p,.section-title,.section-copy{white-space:normal!important;overflow-wrap:anywhere}
.hero-trust{display:flex;flex-wrap:wrap;gap:.45rem;margin-top:1rem}.hero-trust span{display:inline-flex;align-items:center;gap:.36rem;min-height:31px;padding:.3rem .62rem;border:1px solid rgba(188,197,235,.16);border-radius:999px;background:rgba(255,255,255,.045);color:#cbd3e8;font-size:.71rem;font-weight:650}.hero-trust span::before{content:"";width:6px;height:6px;border-radius:50%;background:#31cfab}.hero-trust .attention::before{background:#f4b95f}
.mirror-visual{position:relative;min-height:182px;display:grid;grid-template-columns:1fr 76px 1fr;align-items:center}.mirror-node{position:relative;z-index:2;min-height:104px;padding:.85rem;border:1px solid rgba(184,199,243,.18);border-radius:18px;background:linear-gradient(145deg,rgba(255,255,255,.09),rgba(255,255,255,.025));box-shadow:inset 0 1px 0 rgba(255,255,255,.09);backdrop-filter:blur(10px)}.mirror-node span{display:block;color:#8390af;font-size:.63rem;letter-spacing:.11em;text-transform:uppercase;font-weight:750}.mirror-node b{display:block;color:#f8f9ff;font-size:.9rem;line-height:1.35;margin:.28rem 0 .52rem}.mirror-bars{display:grid;gap:.28rem}.mirror-bars i{height:3px;border-radius:4px;background:linear-gradient(90deg,rgba(34,211,238,.85),rgba(101,88,245,.25));box-shadow:0 0 9px rgba(34,211,238,.25)}.mirror-bars i:nth-child(2){width:73%}.mirror-bars i:nth-child(3){width:88%}.mirror-node:last-child .mirror-bars i{margin-left:auto;background:linear-gradient(90deg,rgba(101,88,245,.28),rgba(157,142,255,.85))}
.mirror-axis{position:relative;z-index:3;display:grid;place-items:center;width:58px;height:138px;margin:auto;border-left:1px solid rgba(93,228,241,.35);border-right:1px solid rgba(146,126,255,.35);border-radius:28px;background:linear-gradient(180deg,rgba(17,184,210,.08),rgba(101,88,245,.13));box-shadow:0 0 26px rgba(63,196,226,.12)}.mirror-axis b{display:grid;place-items:center;width:38px;height:38px;border-radius:13px;background:linear-gradient(145deg,#2ec8dc,#6d5ff8);color:#fff;font-size:.9rem;box-shadow:0 0 26px rgba(73,202,227,.32)}.mirror-axis::before,.mirror-axis::after{content:"";position:absolute;left:50%;width:7px;height:7px;border-radius:50%;background:#55dbe8;box-shadow:0 0 12px #55dbe8}.mirror-axis::before{top:14px;animation:axis-pulse 3.2s ease-in-out infinite}.mirror-axis::after{bottom:14px;background:#8e7cff;box-shadow:0 0 12px #8e7cff;animation:axis-pulse 3.2s 1.6s ease-in-out infinite}
.evidence-beam{position:absolute;left:15%;right:15%;height:1px;z-index:1;background:linear-gradient(90deg,transparent,#35cfdf,#8b78ff,transparent)}.evidence-beam.forward{top:41%;animation:beam-forward 4.8s linear infinite}.evidence-beam.reverse{bottom:39%;transform:scaleX(-1);animation:beam-forward 4.8s 2.4s linear infinite}.evidence-beam::after{content:"";position:absolute;top:-3px;width:7px;height:7px;border-radius:50%;background:#fff;box-shadow:0 0 13px #41d8e7;animation:particle-forward 4.8s linear infinite}

/* Trust ribbon and headings */
.notice{display:grid;grid-template-columns:32px 1fr auto;gap:.7rem;align-items:center;border:1px solid var(--mr-line);background:rgba(255,255,255,.76);color:var(--mr-body);border-radius:16px;padding:.68rem .78rem;margin:.2rem 0 1rem;box-shadow:0 7px 25px rgba(27,33,68,.04);font-size:.82rem;line-height:1.45}.notice::after{content:"Session-only";justify-self:end;padding:.25rem .5rem;border-radius:999px;background:var(--mr-good-soft);color:var(--mr-good);font-size:.68rem;font-weight:750;white-space:nowrap}.notice>div{min-width:0;overflow-wrap:anywhere}.notice-icon{width:28px;height:28px;border:1px solid #b9b2ff;border-radius:9px;display:grid;place-items:center;color:var(--mr-iris-dark);font-size:.72rem;font-weight:800;background:var(--mr-iris-soft)}.notice b{color:var(--mr-ink);font-weight:750}
.section-head{position:relative;margin:1.75rem 0 .9rem;padding-left:.9rem;max-width:980px;min-width:0}.section-head::before{content:"";position:absolute;left:0;top:.1rem;bottom:.1rem;width:3px;border-radius:4px;background:linear-gradient(var(--mr-cyan),var(--mr-iris))}.section-kicker{font-size:.68rem;line-height:1.35;letter-spacing:.12em;text-transform:uppercase;color:var(--mr-iris-dark);font-weight:800;margin-bottom:.25rem}.section-title{font-family:var(--mr-display);font-size:1.62rem;line-height:1.2;letter-spacing:-.032em;color:var(--mr-ink);font-weight:750;margin:0}.section-copy{color:var(--mr-muted);max-width:900px;margin:.42rem 0 0;font-size:.9rem;line-height:1.55}

/* Reciprocal workflow */
.step-grid{counter-reset:mirror-step;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.65rem;margin:.35rem 0 1.2rem}.step-grid.four{grid-template-columns:repeat(4,minmax(0,1fr))}.step{counter-increment:mirror-step;position:relative;padding:.9rem .9rem .9rem 3.35rem;min-height:78px;border:1px solid var(--mr-line);border-radius:16px;background:rgba(255,255,255,.72);box-shadow:0 6px 22px rgba(25,30,60,.035);transition:transform .2s ease,border-color .2s ease,box-shadow .2s ease}.step:hover{transform:translateY(-2px);border-color:#bdb8f7;box-shadow:0 12px 28px rgba(53,49,120,.08)}.step::before{content:attr(data-step);position:absolute;left:.9rem;top:.9rem;width:1.9rem;height:1.9rem;border:0;border-radius:999px;background:linear-gradient(135deg,#35299a,#148da8);color:#fff;display:grid;place-items:center;font-family:var(--mr-font);font-size:.66rem;line-height:1;font-weight:780;font-variant-numeric:tabular-nums;box-shadow:0 5px 14px rgba(55,49,145,.16)}.step:not(:last-child)::after{content:"→";position:absolute;right:-.62rem;top:50%;z-index:2;transform:translateY(-50%);width:1.1rem;height:1.1rem;display:grid;place-items:center;border-radius:50%;background:var(--mr-canvas);color:var(--mr-iris);font-size:.75rem;font-weight:900}.step b{display:block;color:var(--mr-ink);font-size:.84rem;font-weight:750;line-height:1.4}.step span{display:block;color:var(--mr-muted);font-size:.75rem;line-height:1.42;margin-top:.13rem}
[data-testid="stVerticalBlockBorderWrapper"]>div{border-color:var(--mr-line)!important;border-radius:var(--mr-radius)!important;background:var(--mr-surface);box-shadow:0 10px 32px rgba(27,32,68,.045);transition:border-color .2s ease,box-shadow .2s ease,transform .2s ease}[data-testid="stVerticalBlockBorderWrapper"]>div:hover{border-color:#cbc7ed!important;box-shadow:0 16px 40px rgba(33,38,83,.075)}
.st-key-mirror_input_grid{position:relative}.st-key-mirror_input_grid::before{content:"↔";position:absolute;z-index:5;left:50%;top:46%;transform:translate(-50%,-50%);width:40px;height:40px;display:grid;place-items:center;border-radius:14px;background:linear-gradient(145deg,var(--mr-cyan),var(--mr-iris));border:4px solid var(--mr-canvas);color:#fff;font-size:.9rem;font-weight:800;box-shadow:0 8px 23px rgba(70,74,181,.24);pointer-events:none}.st-key-mirror_input_grid::after{content:"additions  →    ←  omissions";position:absolute;z-index:4;left:50%;top:calc(46% + 31px);transform:translateX(-50%);padding:.18rem .38rem;border-radius:999px;background:var(--mr-canvas);color:var(--mr-iris-dark);font-size:.58rem;letter-spacing:.045em;text-transform:uppercase;font-weight:800;white-space:nowrap;pointer-events:none}
.input-panel-title{display:flex;gap:.72rem;align-items:flex-start;margin:.05rem 0 .7rem}.input-panel-title>.input-number{display:inline-grid!important;place-items:center;box-sizing:border-box;flex:0 0 auto;min-width:2.55rem;height:1.9rem;margin:0!important;padding:0 .5rem;border:0;border-radius:999px;background:linear-gradient(135deg,#35299a,#148da8);color:#fff!important;font-family:var(--mr-font)!important;font-size:.62rem;font-weight:780;font-variant-numeric:tabular-nums;letter-spacing:.055em;line-height:1!important;box-shadow:0 5px 14px rgba(55,49,145,.16)}.input-panel-title strong{display:block;color:var(--mr-ink);font-size:.95rem;font-weight:750;line-height:1.35}.input-panel-title>div>span{display:block;color:var(--mr-muted);font-size:.78rem;line-height:1.4;margin-top:.12rem}
.st-key-mirror_demo_console,.st-key-mirror_dataset_mode{margin:.25rem 0 1rem;padding:.85rem 1rem;border:1px solid #d7d4ef;border-left:4px solid var(--mr-iris);border-radius:15px;background:linear-gradient(110deg,#fff,#f5f8ff);box-shadow:0 8px 25px rgba(37,41,84,.05)}.demo-note{display:flex;align-items:flex-start;gap:.65rem;margin:.15rem 0 .2rem;padding:.7rem .78rem;border:1px solid var(--mr-line);border-radius:12px;background:#fff;color:var(--mr-body);font-size:.79rem;line-height:1.48}.demo-note b{color:var(--mr-ink)}.demo-note span{display:inline-grid;place-items:center;flex:0 0 auto;min-width:34px;height:24px;padding:0 .42rem;border-radius:999px;background:var(--mr-iris-soft);color:var(--mr-iris-dark);font-size:.61rem;font-weight:800;letter-spacing:.05em;text-transform:uppercase}
.st-key-mirror_action_console{margin:.7rem 0 1rem;padding:.85rem 1rem;border:1px solid var(--mr-line);border-radius:18px;background:linear-gradient(110deg,#fff,rgba(238,236,255,.65));box-shadow:0 10px 30px rgba(34,37,78,.05)}.st-key-mirror_upload_bento{position:relative}.schema-list{display:grid;gap:.42rem;margin:.7rem 0 .85rem}.schema-row{display:flex;align-items:flex-start;justify-content:space-between;gap:.75rem;padding:.5rem .58rem;border:1px solid var(--mr-line);border-radius:10px;background:var(--mr-soft);font-size:.74rem;line-height:1.4;color:var(--mr-muted)}.schema-row b{color:var(--mr-ink);font-weight:700}.schema-row span{white-space:nowrap;color:var(--mr-iris-dark);font-weight:750}.empty{position:relative;overflow:hidden;border:1px dashed #b7b4d4;background:linear-gradient(135deg,#fff,#f6f5ff);border-radius:18px;padding:1.55rem;text-align:center;color:var(--mr-muted);font-size:.88rem;line-height:1.55}.empty::before{content:"";position:absolute;width:90px;height:90px;right:-34px;top:-45px;border-radius:50%;background:rgba(101,88,245,.08)}.empty b{color:var(--mr-ink);font-weight:750}

/* Audit decision and trace cards */
.risk-hero{position:relative;overflow:hidden;display:grid;grid-template-columns:1fr 205px;gap:1.35rem;align-items:center;background:linear-gradient(130deg,#fff,#f8f8ff);border:1px solid var(--mr-line);border-top:4px solid var(--mr-iris);border-radius:22px;padding:1.35rem 1.4rem;box-shadow:var(--mr-shadow);margin:.65rem 0 1rem}.risk-hero::after{content:"";position:absolute;right:-70px;bottom:-90px;width:230px;height:230px;border:1px solid rgba(101,88,245,.12);border-radius:50%;box-shadow:0 0 0 28px rgba(101,88,245,.035),0 0 0 60px rgba(17,184,210,.025);pointer-events:none}.risk-hero.priority-high{border-top-color:var(--mr-bad)}.risk-hero.priority-review{border-top-color:#b9780a}.risk-hero.priority-routine{border-top-color:var(--mr-good)}.risk-hero .small{font-size:.68rem;line-height:1.35;letter-spacing:.12em;text-transform:uppercase;color:var(--mr-iris-dark);font-weight:800}.risk-hero h2{font-family:var(--mr-display);font-size:1.62rem;line-height:1.22;letter-spacing:-.035em;color:var(--mr-ink);font-weight:750;margin:.27rem 0 .36rem}.risk-hero .copy{color:var(--mr-muted);font-size:.87rem;line-height:1.52;max-width:820px}.risk-score{position:relative;z-index:1;min-height:142px;display:flex;flex-direction:column;justify-content:center;padding:1rem;border:1px solid #d8d5f4;border-radius:18px;background:rgba(255,255,255,.86);box-shadow:inset 0 0 0 7px rgba(101,88,245,.04)}.risk-score span{display:block;color:var(--mr-muted);font-size:.65rem;line-height:1.35;text-transform:uppercase;letter-spacing:.075em;font-weight:800}.risk-score strong{display:block;color:var(--mr-ink);font-size:2rem;line-height:1.1;font-weight:780;letter-spacing:-.045em;margin:.25rem 0}.risk-score small{display:block;color:var(--mr-muted);font-size:.69rem;line-height:1.38}
.direction-lanes{display:grid;grid-template-columns:1fr 1fr;gap:.6rem;margin:.1rem 0 1rem}.direction-lane{position:relative;overflow:hidden;padding:.72rem .82rem;border:1px solid var(--mr-line);border-radius:14px;background:#fff}.direction-lane span{display:block;color:var(--mr-muted);font-size:.66rem;text-transform:uppercase;letter-spacing:.08em;font-weight:800}.direction-lane b{display:block;color:var(--mr-ink);font-size:.9rem;margin-top:.18rem}.direction-lane::after{content:"";position:absolute;bottom:0;height:3px;width:54%;background:linear-gradient(90deg,var(--mr-cyan),var(--mr-iris))}.direction-lane:first-child::after{left:0}.direction-lane:last-child::after{right:0;transform:scaleX(-1)}
.kpi-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.7rem;margin:.7rem 0 1.3rem}.kpi{position:relative;overflow:hidden;background:var(--mr-surface);border:1px solid var(--mr-line);border-radius:16px;padding:.92rem 1rem;min-height:108px;transition:transform .2s ease,box-shadow .2s ease}.kpi:hover{transform:translateY(-2px);box-shadow:0 13px 30px rgba(31,36,80,.07)}.kpi::after{content:"";position:absolute;right:-18px;top:-24px;width:55px;height:55px;border:1px solid rgba(101,88,245,.13);border-radius:50%}.kpi .label{color:var(--mr-muted);font-size:.66rem;line-height:1.35;letter-spacing:.07em;text-transform:uppercase;font-weight:800}.kpi .value{color:var(--mr-ink);font-size:1.48rem;font-weight:780;line-height:1.25;margin:.2rem 0}.kpi .note{color:var(--mr-muted);font-size:.73rem;line-height:1.4}
.claim-card{position:relative;background:var(--mr-surface);border:1px solid var(--mr-line);border-left-width:4px;border-radius:16px;padding:1.05rem 1.1rem;margin:.8rem 0 .34rem;box-shadow:0 8px 25px rgba(28,33,69,.04);overflow-wrap:anywhere;transition:transform .2s ease,box-shadow .2s ease}.claim-card:hover{transform:translateY(-1px);box-shadow:0 14px 34px rgba(28,33,69,.075)}.claim-supported{border-left-color:var(--mr-good)}.claim-contradicted{border-left-color:var(--mr-bad)}.claim-unsupported{border-left-color:#b16b00}.claim-review{border-left-color:var(--mr-iris)}.claim-card-top{display:flex;justify-content:space-between;gap:.65rem;align-items:flex-start}.claim-index{font-size:.68rem;line-height:1.35;letter-spacing:.085em;color:var(--mr-muted);font-weight:800}.status-badge,.risk-pill{display:inline-flex;align-items:center;padding:.27rem .58rem;border-radius:999px;font-size:.72rem;line-height:1.4;font-weight:750}.status-supported{color:#0e6147;background:var(--mr-good-soft)}.status-contradicted{color:#8d2020;background:var(--mr-bad-soft)}.status-unsupported{color:#734500;background:var(--mr-warn-soft)}.status-review{color:#3f319f;background:var(--mr-iris-soft)}.risk-critical,.risk-high{color:#8d2020;background:var(--mr-bad-soft)}.risk-medium{color:#734500;background:var(--mr-warn-soft)}.risk-low{color:#0e6147;background:var(--mr-good-soft)}.risk-unspecified{color:#4e576f;background:#eceef4}.claim-text{font-size:.96rem;line-height:1.62;color:var(--mr-ink);margin:.78rem 0;overflow-wrap:anywhere}.claim-meta{display:flex;gap:.65rem 1rem;flex-wrap:wrap;color:var(--mr-muted);font-size:.76rem;line-height:1.45}.evidence-panel{background:linear-gradient(135deg,var(--mr-soft),#f8f9fc);border:1px solid var(--mr-line);border-radius:14px;padding:.95rem 1rem;overflow-wrap:anywhere}.evidence-kicker,.rationale-label{font-size:.67rem;line-height:1.35;letter-spacing:.08em;text-transform:uppercase;color:var(--mr-ink);font-weight:800}.evidence-panel blockquote{margin:.55rem 0;border-left:3px solid var(--mr-cyan);padding:.58rem .78rem;background:var(--mr-surface);border-radius:0 10px 10px 0;color:var(--mr-body);line-height:1.56;overflow-wrap:anywhere}.evidence-score,.rationale-text{color:var(--mr-muted);font-size:.79rem;line-height:1.5}.evidence-highlight{background:#cff5f7;padding:.05rem .14rem;border-radius:4px}

/* Bento research and readiness surfaces */
.dataset-card,.research-card,.benchmark-card{position:relative;overflow:hidden;height:100%;background:var(--mr-surface);border:1px solid var(--mr-line);border-radius:18px;padding:1.05rem;box-shadow:0 9px 28px rgba(29,34,70,.045);transition:transform .2s ease,border-color .2s ease,box-shadow .2s ease}.dataset-card:hover,.research-card:hover,.benchmark-card:hover{transform:translateY(-3px);border-color:#c4bef6;box-shadow:0 18px 42px rgba(37,42,92,.09)}.dataset-card::after,.research-card::after,.benchmark-card::after{content:"";position:absolute;right:-25px;bottom:-32px;width:80px;height:80px;border:1px solid rgba(101,88,245,.12);border-radius:50%}.dataset-card .badge{display:inline-block;border-radius:999px;background:var(--mr-iris-soft);color:var(--mr-iris-dark);padding:.25rem .52rem;font-size:.65rem;line-height:1.35;font-weight:800;text-transform:uppercase;letter-spacing:.06em}.dataset-card h3,.research-card h3{color:var(--mr-ink);font-size:1rem;font-weight:750;margin:.64rem 0 .28rem}.dataset-card p,.research-card p{font-size:.82rem;color:var(--mr-muted);line-height:1.55;margin-bottom:0}.research-summary{display:grid;grid-template-columns:1.1fr .9fr 1fr;gap:.72rem;margin:.55rem 0 1.25rem}.research-card:nth-child(1){background:linear-gradient(145deg,#fff,#f0fbfc)}.research-card:nth-child(2){transform:translateY(12px);background:linear-gradient(145deg,#fff,#f5f3ff)}.research-card:nth-child(2):hover{transform:translateY(9px)}.research-card span{display:block;color:var(--mr-iris-dark);font-size:.66rem;line-height:1.35;text-transform:uppercase;letter-spacing:.075em;font-weight:800}
.readiness-hero{position:relative;overflow:hidden;display:flex;align-items:center;justify-content:space-between;gap:1.25rem;background:linear-gradient(120deg,#fff,#ecfaf5);border:1px solid #bde3d5;border-radius:20px;padding:1.15rem 1.2rem;margin:.5rem 0 .85rem;box-shadow:var(--mr-shadow)}.readiness-hero::before{content:"";position:absolute;right:17%;top:-45px;width:110px;height:110px;border:1px solid rgba(21,156,134,.15);border-radius:50%;box-shadow:0 0 0 23px rgba(21,156,134,.035)}.readiness-hero.attention{background:linear-gradient(120deg,#fff,#fff7e8);border-color:#ecd097}.readiness-hero h3{color:var(--mr-ink);font-size:1.12rem;line-height:1.4;font-weight:750;margin:0}.readiness-hero p{color:var(--mr-muted);font-size:.81rem;line-height:1.45;margin:.22rem 0 0}.readiness-badge{position:relative;white-space:nowrap;color:#0e6147;background:var(--mr-good-soft);border:1px solid #bce0d2;border-radius:999px;padding:.42rem .68rem;font-size:.72rem;font-weight:800}.readiness-hero.attention .readiness-badge{color:#734500;background:var(--mr-warn-soft);border-color:#ecd097}.check-list{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.58rem;margin:.55rem 0 1rem}.check-row{display:flex;align-items:center;justify-content:space-between;gap:.8rem;min-height:66px;background:var(--mr-surface);border:1px solid var(--mr-line);border-radius:15px;padding:.72rem .8rem;color:var(--mr-ink);font-size:.79rem;line-height:1.4;font-weight:700;box-shadow:0 6px 20px rgba(29,34,70,.035)}.check-status{display:inline-flex;align-items:center;gap:.35rem;color:#0e6147;font-size:.7rem;white-space:nowrap}.check-status.missing{color:#734500}.check-icon{width:23px;height:23px;border-radius:8px;background:var(--mr-good-soft);display:grid;place-items:center;font-size:.7rem}.check-status.missing .check-icon{background:var(--mr-warn-soft)}
.benchmark-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.72rem;margin:.55rem 0 1rem}.benchmark-card:nth-child(2n){background:linear-gradient(145deg,#fff,#f7f6ff)}.benchmark-card .question{color:var(--mr-iris-dark);font-size:.65rem;line-height:1.35;text-transform:uppercase;letter-spacing:.075em;font-weight:800}.benchmark-card h3{color:var(--mr-ink);font-size:.9rem;line-height:1.4;font-weight:750;margin:.4rem 0 .3rem}.benchmark-card strong{display:block;color:var(--mr-ink);font-size:1.62rem;line-height:1.2;font-weight:780;letter-spacing:-.04em;margin:.3rem 0}.benchmark-card p{color:var(--mr-muted);font-size:.75rem;line-height:1.45;margin:0}.flow{position:relative;display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:.65rem;margin:1rem 0}.flow::before{content:"";position:absolute;left:5%;right:5%;top:50%;height:1px;background:linear-gradient(90deg,var(--mr-cyan),var(--mr-iris),var(--mr-cyan));z-index:0}.flow-node{position:relative;z-index:1;background:var(--mr-surface);border:1px solid var(--mr-line);border-radius:16px;padding:.92rem .8rem;text-align:left;min-height:104px;box-shadow:0 8px 24px rgba(29,34,70,.05)}.flow-node:nth-child(3){transform:translateY(-10px);border-color:#bcb4ff;background:linear-gradient(145deg,#fff,#efedff);box-shadow:0 15px 35px rgba(72,59,178,.13)}.flow-node::before{content:"";display:block;width:28px;height:4px;border-radius:4px;background:linear-gradient(90deg,var(--mr-cyan),var(--mr-iris));margin-bottom:.62rem}.flow-node b{display:block;color:var(--mr-ink);font-size:.82rem;line-height:1.4;font-weight:750}.flow-node span{display:block;font-size:.73rem;line-height:1.45;color:var(--mr-muted);margin-top:.16rem}

/* Native Streamlit controls */
.stButton>button,.stDownloadButton>button{border-radius:13px;min-height:46px;font-family:var(--mr-font);font-weight:700;border-color:var(--mr-line-strong);color:var(--mr-ink);background:var(--mr-surface);transition:transform .18s ease,box-shadow .18s ease,border-color .18s ease,background .18s ease}.stButton>button:hover,.stDownloadButton>button:hover{transform:translateY(-1px);border-color:var(--mr-iris);color:var(--mr-iris-dark);background:#faf9ff;box-shadow:0 9px 22px rgba(67,57,158,.1)}.stButton>button[kind="primary"]{position:relative;background:linear-gradient(105deg,var(--mr-iris-dark),var(--mr-iris) 56%,#5485ef);color:#fff;border:1px solid #5b4ce1;box-shadow:0 12px 28px rgba(73,57,184,.25)}.stButton>button[kind="primary"]:hover{background:linear-gradient(105deg,#3f309f,#584ae3 56%,#4375df);color:#fff;border-color:#4939b8;box-shadow:0 16px 34px rgba(73,57,184,.31)}.stButton>button:disabled,.stDownloadButton>button:disabled{opacity:.55;box-shadow:none;transform:none}
.stTextArea label,.stFileUploader label,.stCheckbox label,.stSelectbox label,.stMultiSelect label{color:var(--mr-ink)!important;font-size:.84rem!important;font-weight:700!important}.stTextArea textarea,.stTextInput input{border-radius:13px;border-color:var(--mr-line-strong);line-height:1.58;background:#fbfbfe;color:var(--mr-ink);font-family:var(--mr-font);font-size:.91rem}.stTextArea textarea::placeholder,.stTextInput input::placeholder{color:#747c91;opacity:1}.stTextArea textarea:focus,.stTextInput input:focus{border-color:var(--mr-focus);box-shadow:0 0 0 4px rgba(82,69,215,.13)}
[data-testid="stFileUploaderDropzone"]{position:relative;background:linear-gradient(135deg,#f9f9ff,#edf9fb);border:1px dashed #9f9acb;border-radius:16px;min-height:150px}[data-testid="stFileUploaderDropzone"]::after{content:"Secure session-memory intake";position:absolute;right:.8rem;bottom:.6rem;color:var(--mr-iris-dark);font-size:.62rem;text-transform:uppercase;letter-spacing:.06em;font-weight:800}[data-testid="stFileUploaderDropzone"] button{border-radius:11px}
[data-testid="stMetric"]{position:relative;overflow:hidden;background:var(--mr-surface);border:1px solid var(--mr-line);border-radius:16px;padding:.86rem .94rem;min-height:100px;box-shadow:0 7px 23px rgba(29,34,70,.04)}[data-testid="stMetric"]::after{content:"";position:absolute;right:-20px;bottom:-24px;width:65px;height:65px;border:1px solid rgba(101,88,245,.12);border-radius:50%}[data-testid="stMetricLabel"] p{color:var(--mr-muted);font-size:.74rem;font-weight:700}[data-testid="stMetricValue"]{color:var(--mr-ink);font-weight:780;letter-spacing:-.035em}
[data-testid="stDataFrame"],[data-testid="stTable"]{border:1px solid var(--mr-line);border-radius:14px;overflow:hidden;background:var(--mr-surface);box-shadow:0 8px 25px rgba(29,34,70,.04)}[data-baseweb="tab-list"]{gap:.35rem;border-bottom:1px solid var(--mr-line);background:#fff;padding:.35rem;border-radius:15px 15px 0 0}[data-baseweb="tab"]{height:44px;padding:0 .82rem;color:var(--mr-muted);font-weight:700;border-radius:10px}[data-baseweb="tab"][aria-selected="true"]{color:var(--mr-iris-dark);background:linear-gradient(135deg,var(--mr-iris-soft),var(--mr-cyan-soft))}
div[data-testid="stExpander"]{background:var(--mr-surface);border-color:var(--mr-line);border-radius:14px}[data-testid="stAlert"]{border-radius:14px;font-size:.84rem;line-height:1.5}[data-testid="stCaptionContainer"]{color:var(--mr-muted);font-size:.78rem;line-height:1.5}[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li{line-height:1.62}hr{border-color:var(--mr-line)!important}
button:focus-visible,input:focus-visible,textarea:focus-visible,[tabindex]:focus-visible{outline:3px solid rgba(82,69,215,.42)!important;outline-offset:3px!important}

@keyframes brand-scan{0%{transform:translateX(-500%) rotate(24deg)}100%{transform:translateX(700%) rotate(24deg)}}
@keyframes mirror-sweep{0%,100%{transform:translateX(-180%) rotate(20deg);opacity:0}20%,75%{opacity:1}55%{transform:translateX(430%) rotate(20deg)}}
@keyframes health-pulse{0%,100%{box-shadow:0 0 0 4px rgba(53,211,167,.1)}50%{box-shadow:0 0 0 8px rgba(53,211,167,0)}}
@keyframes axis-pulse{0%,100%{transform:scale(.75);opacity:.5}50%{transform:scale(1.15);opacity:1}}
@keyframes beam-forward{0%,100%{opacity:.2}50%{opacity:.85}}
@keyframes particle-forward{0%{left:0;opacity:0}10%{opacity:1}90%{opacity:1}100%{left:100%;opacity:0}}

@media(max-width:1100px){.block-container{padding-left:1.35rem;padding-right:1.35rem}.app-head{grid-template-columns:1fr .72fr;gap:1.4rem}.mirror-brandcopy span{max-width:410px}.kpi-grid,.benchmark-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.check-list{grid-template-columns:repeat(2,minmax(0,1fr))}.flow{grid-template-columns:repeat(3,minmax(0,1fr))}.flow::before{display:none}}
@media(max-width:820px){.mirror-brandbar{align-items:flex-start}.mirror-command-meta .command-chip:last-child{display:none}.app-head{grid-template-columns:1fr}.mirror-visual{min-height:150px}.app-head h1{max-width:650px}.research-summary{grid-template-columns:1fr 1fr}.research-card:nth-child(2){transform:none}.step-grid.four{grid-template-columns:repeat(2,minmax(0,1fr))}.step-grid.four .step:nth-child(2)::after{display:none}}
@media(max-width:760px){
  html,body,#root,.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"]{max-width:100vw!important;min-width:0!important;overflow-x:hidden!important}.block-container{box-sizing:border-box!important;width:100%!important;max-width:100%!important;min-width:0!important;padding:3.65rem .82rem 3.5rem;overflow-x:hidden!important}
  [data-testid="stHorizontalBlock"]{flex-direction:column!important;gap:.8rem!important;min-width:0!important;width:100%!important}[data-testid="column"]{box-sizing:border-box!important;flex:1 1 100%!important;width:100%!important;min-width:0!important;max-width:100%!important}
  .mirror-brandbar{display:block;min-height:0;padding:.85rem;border-radius:18px 18px 7px 7px}.mirror-mark{width:46px;height:46px;border-radius:14px}.mirror-mark::before,.mirror-mark::after{top:10px}.mirror-mark::before{left:10px}.mirror-mark::after{right:10px}.mirror-brandcopy strong{font-size:1.12rem}.mirror-brandcopy span{font-size:.67rem;max-width:270px}.mirror-command-meta{justify-content:flex-start;margin:.75rem 0 0}.command-chip{font-size:.66rem;min-height:31px}
  .st-key-mirror_navigation{box-sizing:border-box;width:100%;max-width:100%;padding:.28rem;border-radius:13px;overflow:hidden}
  .st-key-mirror_navigation [data-testid="stElementContainer"],.st-key-mirror_navigation [data-testid="stRadio"]{box-sizing:border-box;width:100%!important;max-width:100%!important;min-width:0!important}
  .st-key-mirror_navigation div[role="radiogroup"]{display:grid!important;grid-template-columns:repeat(6,minmax(0,1fr))!important;grid-auto-flow:row;align-items:stretch;width:100%!important;max-width:100%!important;min-width:0!important;gap:.22rem;overflow:visible!important;padding:0!important}
  .st-key-mirror_navigation div[role="radiogroup"] label{grid-column:span 2;min-width:0!important;min-height:44px;padding:.6rem .58rem}
  .st-key-mirror_navigation div[role="radiogroup"] label:nth-of-type(n+4){grid-column:span 3}
  .st-key-mirror_navigation div[role="radiogroup"] p{font-size:.75rem}
  .app-head,.notice,.step-grid,.section-head{box-sizing:border-box!important;width:100%!important;max-width:100%!important;min-width:0!important}.app-head{display:block;min-height:0;margin-top:.85rem;padding:1.35rem 1.05rem;border-radius:22px}.app-head h1{font-size:2.05rem;letter-spacing:-.045em;max-width:100%;word-break:normal!important;overflow-wrap:anywhere!important}.app-head p{font-size:.87rem;line-height:1.5}.hero-trust{margin-top:.8rem}.mirror-visual{margin-top:1.1rem;grid-template-columns:1fr 54px 1fr;min-height:134px}.mirror-node{min-height:90px;padding:.65rem}.mirror-node b{font-size:.75rem}.mirror-axis{width:42px;height:112px}.mirror-axis b{width:31px;height:31px;border-radius:10px}.notice{grid-template-columns:28px 1fr;font-size:.77rem}.notice::after{display:none}
  .step-grid,.step-grid.four,.kpi-grid,.benchmark-grid,.research-summary,.check-list,.flow,.direction-lanes{grid-template-columns:1fr}.step{min-height:70px}.step:not(:last-child)::after{content:"↓";right:auto;left:50%;top:auto;bottom:-.62rem;transform:translateX(-50%)}.research-card:nth-child(2),.flow-node:nth-child(3){transform:none}.readiness-hero{display:block}.readiness-badge{display:inline-block;margin-top:.7rem}.risk-hero{grid-template-columns:1fr;padding:1rem;border-radius:18px}.risk-score{min-height:0}.risk-score strong{font-size:1.65rem}.section-title{font-size:1.38rem;max-width:100%;word-break:normal}.section-head{margin-top:1.45rem}[data-baseweb="tab-list"]{overflow-x:auto;scrollbar-width:thin}.claim-card-top{display:block}.claim-card-top>div:last-child{margin-top:.5rem}.st-key-mirror_input_grid::before,.st-key-mirror_input_grid::after{display:none}
}
@media(max-width:520px){
  .st-key-mirror_navigation div[role="radiogroup"]{grid-template-columns:repeat(4,minmax(0,1fr))!important}
  .st-key-mirror_navigation div[role="radiogroup"] label:nth-of-type(n){grid-column:span 2!important}
  .st-key-mirror_navigation div[role="radiogroup"] label:nth-of-type(5){grid-column:2/span 2!important}
}
@media(max-width:430px){.block-container{padding-left:.62rem;padding-right:.62rem}.mirror-brandcopy span{max-width:215px}.command-chip:nth-child(2){display:none}.mirror-node span{font-size:.54rem}.mirror-node b{font-size:.68rem}.notice{font-size:.74rem}.kpi{min-height:0}.app-head h1{font-size:1.82rem}}
@media(max-width:330px){.st-key-mirror_navigation div[role="radiogroup"]{grid-template-columns:1fr!important}.st-key-mirror_navigation div[role="radiogroup"] label:nth-of-type(n){grid-column:1/-1!important}}
@media(forced-colors:active){.st-key-mirror_navigation div[role="radiogroup"] label:has(input:focus-visible){outline:2px solid Highlight;outline-offset:-2px;box-shadow:none}}
@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important;transition:none!important;animation:none!important}.step:hover,.dataset-card:hover,.research-card:hover,.benchmark-card:hover,.claim-card:hover,.stButton>button:hover,.stDownloadButton>button:hover{transform:none!important}}
</style>
"""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _release_fingerprint(path: Path) -> tuple[tuple[str, int, int], ...]:
    """Return a cheap cache key that changes when any sealed asset changes."""

    manifest = _read_json(path)
    entries: list[tuple[str, int, int]] = []
    if path.is_file():
        stat = path.stat()
        entries.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    for value in manifest.get("assets", {}).values():
        if not isinstance(value, dict):
            continue
        configured = Path(str(value.get("path", "")))
        target = configured if configured.is_absolute() else PROJECT_ROOT / configured
        if value.get("kind") == "directory" and target.is_dir():
            for child in target.rglob("*"):
                if not child.is_file() or "__pycache__" in child.parts:
                    continue
                stat = child.stat()
                entries.append((str(child.resolve()), stat.st_size, stat.st_mtime_ns))
            continue
        try:
            stat = target.stat()
        except OSError:
            entries.append((str(target.resolve()), -1, -1))
        else:
            entries.append((str(target.resolve()), stat.st_size, stat.st_mtime_ns))
    return tuple(sorted(entries))


@st.cache_data(show_spinner=False)
def _validated_release(
    manifest_path: str,
    fingerprint: tuple[tuple[str, int, int], ...],
) -> dict[str, Any]:
    del fingerprint
    # Enforce the application's declared schema even if a long-lived cloud
    # worker retained an older validator module before this script restarted.
    release_runtime.RELEASE_SCHEMA = RELEASE_SCHEMA
    result = release_runtime.validate_release_manifest(manifest_path, PROJECT_ROOT)
    try:
        observed_schema = json.loads(Path(manifest_path).read_text(encoding="utf-8")).get(
            "schema"
        )
    except (OSError, TypeError, ValueError):
        return result
    schema_error = "Unsupported release manifest schema."
    if observed_schema == RELEASE_SCHEMA and schema_error in result.get("errors", []):
        remaining_errors = [
            error for error in result["errors"] if error != schema_error
        ]
        result["errors"] = remaining_errors
        result["valid"] = not remaining_errors
    return result


def _configured_artifact() -> Path:
    configured = os.getenv(ENV_ARTIFACT, "").strip()
    path = Path(configured) if configured else DEFAULT_ARTIFACT
    return path if path.is_absolute() else PROJECT_ROOT / path


def artifact_health() -> dict[str, Any]:
    artifact = _configured_artifact().resolve()
    config_path = artifact / "verifier_config.json"
    fusion_path = artifact / "fusion_model.pkl"
    calibrator_path = DEFAULT_CALIBRATOR.resolve()
    config = _read_json(config_path)
    reference_value = str(config.get("nli_model", "")).replace("\\", "/")
    model_reference = Path(reference_value) if reference_value else None
    if model_reference is not None and not model_reference.is_absolute():
        model_reference = PROJECT_ROOT / model_reference
    model_ready = bool(
        model_reference
        and (model_reference / "config.json").is_file()
        and (model_reference / "model.safetensors").is_file()
        and (model_reference / "tokenizer_config.json").is_file()
        and (
            (model_reference / "tokenizer.json").is_file()
            or (model_reference / "vocab.txt").is_file()
        )
    )
    calibrator_valid = False
    calibrator_metadata: dict[str, Any] = {}
    try:
        calibrator = ReportRiskCalibrator.load(calibrator_path)
        calibrator_metadata = dict(calibrator.metadata)
        calibrator_valid = bool(
            calibrator.is_fitted
            and calibrator.method
            in {
                "monotonic_raw_isotonic",
                "nonnegative_two_feature_logistic",
                "standardized_l2_logistic",
            }
            and calibrator_metadata.get("fit_granularity") == "complete_report"
            and calibrator_metadata.get("review_threshold") is not None
            and calibrator_metadata.get("high_priority_threshold") is not None
        )
    except Exception:  # noqa: BLE001 - corrupt joblib payloads must fail readiness safely
        calibrator_valid = False
    release = _validated_release(
        str(RELEASE_MANIFEST),
        _release_fingerprint(RELEASE_MANIFEST),
    )
    ledger = _read_json(EVALUATION_SUMMARY)
    ledger_valid = bool(
        ledger.get("schema") == "mirror-rad.evaluation.v1"
        and ledger.get("project") == PRODUCT_NAME
        and ledger.get("completion_status") == "complete"
    )
    checks = {
        "verifier_configuration": config_path.is_file(),
        "fusion_model": fusion_path.is_file(),
        "clinical_nli_checkpoint": model_ready,
        "complete_report_router": calibrator_valid,
        "evaluation_ledger": ledger_valid,
        "sealed_release_manifest": bool(release["valid"]),
    }
    return {
        "artifact": artifact,
        "calibrator": calibrator_path,
        "config": config,
        "calibrator_metadata": calibrator_metadata,
        "release": release,
        "checks": checks,
        "ready": all(checks.values()),
    }


@st.cache_resource(show_spinner=False)
def load_runtime(
    artifact_path: str,
    calibrator_path: str,
    device: str,
    release_id: str,
) -> tuple[Any, Any]:
    del release_id  # Included in the cache key so sealed releases cannot reuse stale objects.
    verifier = RadTraceVerifier.load_artifact(artifact_path, device_override=device)
    # Loading is intentionally forced here. The transformer backend is lazy by
    # design, so inspecting ``load_error`` alone would not prove that the
    # checkpoint and tokenizer can actually initialize.
    available = getattr(verifier.nli_predictor, "available", True)
    load_error = getattr(verifier.nli_predictor, "load_error", None)
    if not available or load_error:
        detail = load_error or "backend reported unavailable"
        raise RuntimeError(f"Clinical NLI checkpoint could not be initialized: {detail}")
    if verifier.fusion_model is None or not verifier.fusion_model.is_fitted:
        raise RuntimeError("The selected inference artifact does not contain a fitted fusion model.")
    calibrator = ReportRiskCalibrator.load(calibrator_path)
    return verifier, BidirectionalAuditor(verifier, calibrator=calibrator)


def require_runtime() -> tuple[Any | None, Any | None]:
    health = artifact_health()
    if not health["ready"]:
        for key in (
            "single_audit",
            "single_audit_ms",
            "single_audit_input_hash",
            "batch_run",
        ):
            st.session_state.pop(key, None)
        st.error("The verified neural runtime is incomplete. Open System Readiness for exact checks.")
        return None, None
    device = os.getenv(ENV_DEVICE, "auto").strip().lower() or "auto"
    release_id = str(health["release"].get("release_id") or "unsealed")
    if st.session_state.get("runtime_release_id") != release_id:
        st.session_state.pop("single_audit", None)
        st.session_state.pop("single_audit_ms", None)
        st.session_state.pop("batch_run", None)
        st.session_state.runtime_release_id = release_id
    try:
        with st.spinner("Loading the verified clinical NLI runtime…"):
            return load_runtime(
                str(health["artifact"]),
                str(health["calibrator"]),
                device,
                release_id,
            )
    except Exception as exc:  # noqa: BLE001 - application boundary
        st.error("The neural runtime failed closed; no heuristic result was substituted.")
        with st.expander("Technical setup details"):
            st.code(f"{type(exc).__name__}: {exc}")
        return None, None


PAGE_SPECS: dict[str, dict[str, str]] = {
    "Single Report": {
        "nav": "Mirror Audit",
        "kicker": "Reciprocal audit · One report",
        "title": "See what changed—and what disappeared",
        "description": (
            "Place the source report and candidate summary on opposite sides of the evidence mirror. "
            "Every alert travels back to the text that supports, conflicts with, or fails to preserve it."
        ),
        "left": "Source evidence",
        "right": "Candidate summary",
    },
    "Dataset Audit": {
        "nav": "Dataset Lab",
        "kicker": "Reciprocal audit · Dataset",
        "title": "Turn report pairs into a review queue",
        "description": (
            "Securely validate, analyze, prioritize, and export a de-identified collection of source-report "
            "and summary pairs through the same two-direction neural engine."
        ),
        "left": "Report pairs",
        "right": "Review queue",
    },
    "Results & Benchmarks": {
        "nav": "Evidence Atlas",
        "kicker": "Scientific evidence · Frozen ledgers",
        "title": "Explore what the evidence actually supports",
        "description": (
            "Navigate claim detection, radiologist-consensus ranking, external transfer, ablations, "
            "calibration, and the boundaries that keep every result honest."
        ),
        "left": "Measured results",
        "right": "Validity limits",
    },
    "Study Design": {
        "nav": "Research Map",
        "kicker": "Research map · Traceable design",
        "title": "Follow the idea from claim to decision",
        "description": (
            "Trace the research problem, mutual-inference architecture, evaluation datasets, "
            "innovation, and responsible-use boundary as one connected system."
        ),
        "left": "Source claims",
        "right": "Summary claims",
    },
    "System Readiness": {
        "nav": "Release Console",
        "kicker": "Control plane · Release integrity",
        "title": "Know exactly what is ready to run",
        "description": (
            "Verify the neural model, learned fusion, review router, evidence ledger, and sealed release "
            "before a single research audit is allowed to execute."
        ),
        "left": "Sealed assets",
        "right": "Live inference",
    },
}


def render_navigation() -> str:
    health = artifact_health()
    release_id = str(health.get("release", {}).get("release_id") or "unsealed")
    status = "System ready" if health["ready"] else "Setup attention needed"
    status_detail = (
        f"Checksum-verified research build {release_id[:12]}; full details are in Release Console."
        if health["ready"]
        else "One or more required research assets needs attention; open Release Console for details."
    )
    status_class = "command-chip" if health["ready"] else "command-chip attention"
    st.markdown(
        '<header class="mirror-brandbar" aria-label="MIRROR-Rad command bar">'
        '<div class="mirror-brand"><div class="mirror-mark" aria-hidden="true"><span></span></div>'
        f'<div class="mirror-brandcopy"><strong>{escape(PRODUCT_NAME)}</strong>'
        f'<span>{escape(PRODUCT_EXPANSION)}</span></div></div>'
        '<div class="mirror-command-meta">'
        f'<span class="{status_class}" title="{escape(status_detail)}">'
        '<span class="health-dot" aria-hidden="true"></span>'
        f'<b>{escape(status)}</b></span>'
        '<span class="command-chip">Session-only research</span>'
        '</div></header>',
        unsafe_allow_html=True,
    )
    with st.container(key="mirror_navigation"):
        page = st.radio(
            "Primary navigation",
            tuple(PAGE_SPECS),
            format_func=lambda value: PAGE_SPECS[value]["nav"],
            label_visibility="collapsed",
            key="primary_workspace",
        )
    return page


def render_shell(page: str) -> None:
    spec = PAGE_SPECS[page]
    ready = artifact_health()["ready"]
    state = "System ready" if ready else "Setup attention required"
    trust_class = "" if ready else ' class="attention"'
    page_id = f"workspace-{hashlib.sha1(page.encode('utf-8')).hexdigest()[:10]}"
    st.html(
        '<header class="app-head"><div class="hero-copy">'
        f'<div class="page-kicker">{escape(spec["kicker"])}</div>'
        f'<h1 id="{page_id}">{escape(spec["title"])}</h1><p>{escape(spec["description"])}</p>'
        '<div class="hero-trust">'
        f'<span{trust_class}>{escape(state)}</span><span>Session-only research</span>'
        '<span>Research workflow · not diagnosis</span></div></div>'
        '<div class="mirror-visual" aria-hidden="true">'
        '<div class="evidence-beam forward"></div><div class="evidence-beam reverse"></div>'
        f'<div class="mirror-node"><span>Input side</span><b>{escape(spec["left"])}</b>'
        '<div class="mirror-bars"><i></i><i></i><i></i></div></div>'
        '<div class="mirror-axis"><b>M</b></div>'
        f'<div class="mirror-node"><span>Reciprocal side</span><b>{escape(spec["right"])}</b>'
        '<div class="mirror-bars"><i></i><i></i><i></i></div></div></div></header>'
    )


def render_notice() -> None:
    st.markdown(
        '<div class="notice"><span class="notice-icon" aria-hidden="true">i</span><div>'
        '<b>Text-consistency research only.</b> The mirror compares supplied text—not radiographs—and '
        'never establishes image truth, diagnosis, or treatment. On an online deployment, inputs are '
        'sent to the hosting service before screening. Never upload PHI or identifiable patient data.</div></div>',
        unsafe_allow_html=True,
    )


def section(kicker: str, title: str, copy: str = "") -> None:
    section_id = f"section-{hashlib.sha1(title.encode('utf-8')).hexdigest()[:10]}"
    description = f'<p class="section-copy">{escape(copy)}</p>' if copy else ""
    st.markdown(
        f'<section class="section-head" aria-labelledby="{section_id}">'
        f'<div class="section-kicker">{escape(kicker)}</div>'
        f'<h2 class="section-title" id="{section_id}">{escape(title)}</h2>'
        f'{description}</section>',
        unsafe_allow_html=True,
    )


def _normalized_direction(audit: Any, review_threshold: float) -> dict[str, Any]:
    return normalize_audit_result(
        audit,
        findings=audit.findings,
        impression=audit.impression,
        review_threshold=review_threshold,
    )


def render_claims(audit: Any, *, direction: str, review_threshold: float) -> None:
    claims = _normalized_direction(audit, review_threshold)["claims"]
    if not claims:
        st.info("No claims were available in this direction.")
        return
    order = {"CONTRADICTED": 0, "UNSUPPORTED": 1, "REVIEW": 2, "ENTAILED": 3}
    indexed = sorted(enumerate(claims, start=1), key=lambda item: (order[item[1]["label"]], item[0]))
    for original_index, claim in indexed:
        st.markdown(claim_card_html(claim, original_index), unsafe_allow_html=True)
        with st.expander(f"Decision trace · claim {original_index}"):
            panel = evidence_panel_html(claim).replace("Model rationale", "Decision trace")
            st.markdown(panel, unsafe_allow_html=True)
            evidence_rows = []
            for evidence in claim.get("evidence", []):
                probabilities = evidence.get("nli_probabilities", {})
                evidence_rows.append(
                    {
                        "Rank": evidence.get("rank"),
                        "Source sentence": evidence.get("text"),
                        "Retrieval similarity": evidence.get("score"),
                        "NLI support": probabilities.get("ENTAILED"),
                        "NLI contradiction": probabilities.get("CONTRADICTED"),
                        "NLI unsupported": probabilities.get("UNSUPPORTED"),
                    }
                )
            if evidence_rows:
                st.markdown("**Complete top-k evidence ledger**")
                st.dataframe(evidence_rows, hide_index=True, width="stretch")
            probabilities = claim.get("probabilities") or {}
            if probabilities:
                raw_label = max(probabilities, key=lambda key: float(probabilities[key]))
                columns = st.columns(3)
                for column, label in zip(columns, ("ENTAILED", "CONTRADICTED", "UNSUPPORTED")):
                    column.metric(LABEL_META[label]["display"], f"{float(probabilities.get(label, 0)):.1%}")
                raw_display = LABEL_META.get(raw_label, {"display": raw_label})["display"]
                st.caption(
                    f"Highest-scoring claim class before abstention: {raw_display} · Displayed decision: "
                    f"{LABEL_META[claim['label']]['display']} · Direction: {direction}"
                )
                st.caption(
                    "Claim-class scores were evaluated and calibrated on the controlled claim benchmark; "
                    "they are not probabilities of image truth, disease, harm, or diagnostic correctness."
                )
            flags = claim.get("rule_flags") or {}
            violations = flags.get("violations", []) if isinstance(flags, dict) else []
            if violations:
                display = ", ".join(str(value).replace("_", " ").title() for value in violations)
                st.markdown(f"**Structured checks:** {display}")
            st.caption("Retrieval similarity locates relevant text; it is not factuality probability.")


def _result_kpis(result: Any, elapsed_ms: float) -> None:
    forward = summarize_claims(_normalized_direction(result.commission_audit, 0.0)["claims"])
    reverse = summarize_claims(_normalized_direction(result.omission_audit, 0.0)["claims"])
    st.markdown(
        f'<div class="kpi-grid"><div class="kpi"><div class="label">Review priority</div>'
        f'<div class="value">{escape(result.review_priority.title())}</div><div class="note">Report review queue</div></div>'
        f'<div class="kpi"><div class="label">Added or conflicting</div><div class="value">{forward["flagged"]}</div>'
        f'<div class="note">of {forward["total"]} summary claims checked</div></div>'
        f'<div class="kpi"><div class="label">Missing source details</div><div class="value">{reverse["flagged"]}</div>'
        f'<div class="note">of {reverse["total"]} source claims checked</div></div>'
        f'<div class="kpi"><div class="label">Analysis time</div><div class="value">{elapsed_ms / 1000:.2f}s</div>'
        '<div class="note">Neural evidence audit</div></div></div>',
        unsafe_allow_html=True,
    )


def _flatten_claim_rows(result: Any, *, case_id: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for direction, audit in (
        ("commission", result.commission_audit),
        ("omission", result.omission_audit),
    ):
        for claim in audit.claims:
            evidence_items = claim.evidence or [None]
            for evidence in evidence_items:
                nli = evidence.nli_probabilities if evidence else {}
                output.append(
                    {
                    "case_id": case_id,
                    "direction": direction,
                    "claim_id": claim.claim_id,
                    "claim": claim.claim,
                    "label": claim.label.value,
                    "confidence": round(claim.confidence, 6),
                    "risk_tier": claim.risk_tier.value,
                    "error_types": "; ".join(value.value for value in claim.error_types),
                    "claim_score_entailed": round(claim.probabilities.get("ENTAILED", 0.0), 6),
                    "claim_score_contradicted": round(
                        claim.probabilities.get("CONTRADICTED", 0.0), 6
                    ),
                    "claim_score_unsupported": round(
                        claim.probabilities.get("UNSUPPORTED", 0.0), 6
                    ),
                    "evidence_rank": evidence.rank if evidence else "",
                    "evidence": evidence.text if evidence else "",
                    "evidence_score": round(evidence.score, 6) if evidence else 0.0,
                    "evidence_nli_entailed": round(nli.get("ENTAILED", 0.0), 6),
                    "evidence_nli_contradicted": round(nli.get("CONTRADICTED", 0.0), 6),
                    "evidence_nli_unsupported": round(nli.get("UNSUPPORTED", 0.0), 6),
                    "decision_trace": claim.explanation,
                    }
                )
    return output


def _runtime_provenance(verifier: Any, auditor: Any) -> dict[str, Any]:
    health = artifact_health()
    assets = health.get("release", {}).get("assets", {})
    asset_hashes = {
        name: value.get("sha256")
        for name, value in assets.items()
        if isinstance(value, dict)
    }
    metadata = dict(getattr(getattr(auditor, "calibrator", None), "metadata", {}))
    package_names = ("torch", "transformers", "streamlit", "scikit-learn")
    versions: dict[str, str | None] = {}
    for name in package_names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "release_id": health.get("release", {}).get("release_id"),
        "asset_sha256": asset_hashes,
        "neural_backend": getattr(verifier.nli_predictor, "name", None),
        "neural_device": getattr(verifier.nli_predictor, "effective_device", None),
        "neural_fallback_used": bool(getattr(verifier.nli_predictor, "load_error", None)),
        "review_router": {
            "method": getattr(getattr(auditor, "calibrator", None), "method", None),
            "fit_granularity": metadata.get("fit_granularity"),
            "fit_report_split_sha256": metadata.get("fit_report_split_sha256"),
            "selection_report_split_sha256": metadata.get("selection_report_split_sha256"),
            "review_threshold": getattr(auditor, "review_threshold", None),
            "high_priority_threshold": getattr(auditor, "high_priority_threshold", None),
            "score_semantics": metadata.get("score_semantics"),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": versions,
        },
    }


def _assert_neural_active(verifier: Any) -> None:
    if not getattr(verifier.nli_predictor, "available", False):
        detail = getattr(verifier.nli_predictor, "load_error", None) or "backend unavailable"
        raise RuntimeError(f"Clinical NLI backend is unavailable: {detail}")


def _safe_cell(value: Any) -> Any:
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _rows_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _safe_cell(value) for key, value in row.items()})
    return buffer.getvalue()


def render_audit_result(
    result: Any,
    *,
    elapsed_ms: float,
    review_threshold: float,
    verifier: Any,
    auditor: Any,
) -> None:
    forward = summarize_claims(_normalized_direction(result.commission_audit, 0.0)["claims"])
    reverse = summarize_claims(_normalized_direction(result.omission_audit, 0.0)["claims"])
    issue_total = int(forward["flagged"]) + int(reverse["flagged"])
    priority_key = str(result.review_priority).strip().upper()
    if priority_key == "HIGH":
        outcome_title = "High-priority review recommended"
        outcome_copy = (
            f"The audit found {issue_total} flagged claim path(s). Review the evidence traces before this "
            "summary is used downstream."
        )
        priority_class = "priority-high"
    elif priority_key == "REVIEW":
        outcome_title = "Review recommended"
        outcome_copy = (
            f"The audit found {issue_total} flagged or uncertain claim path(s). Inspect the source evidence "
            "and model abstentions."
        )
        priority_class = "priority-review"
    else:
        outcome_title = "Routine review queue"
        outcome_copy = (
            f"The audit found {issue_total} flagged claim path(s) at the configured thresholds. This does not "
            "verify the radiograph or establish clinical correctness."
        )
        priority_class = "priority-routine"
    st.markdown(
        f'<section class="risk-hero {priority_class}" aria-label="Audit outcome"><div>'
        f'<div class="small">Audit outcome</div><h2>{escape(outcome_title)}</h2>'
        f'<div class="copy">{escape(outcome_copy)}</div></div>'
        f'<div class="risk-score"><span>Review-routing score</span><strong>{result.aligned_risk * 100:.1f} / 100</strong>'
        '<small>Research queue signal · not a diagnostic probability</small></div></section>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="direction-lanes" role="list" aria-label="Reciprocal evidence directions">'
        '<div class="direction-lane" role="listitem"><span>Summary → Source · additions and conflicts</span>'
        f'<b>{result.commission.score * 100:.1f} / 100 directional risk</b></div>'
        '<div class="direction-lane" role="listitem"><span>Source → Summary · omitted content</span>'
        f'<b>{result.omission.score * 100:.1f} / 100 directional risk</b></div></div>',
        unsafe_allow_html=True,
    )
    _result_kpis(result, elapsed_ms)
    commission_tab, omission_tab = st.tabs(
        ["Added or conflicting claims", "Missing from the summary"]
    )
    with commission_tab:
        st.caption("Commission path · checks every summary claim against the supplied source report.")
        render_claims(result.commission_audit, direction="commission", review_threshold=review_threshold)
    with omission_tab:
        st.caption("Omission path · checks whether each source claim is preserved in the supplied summary.")
        render_claims(result.omission_audit, direction="omission", review_threshold=review_threshold)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    left, right, note = st.columns([1, 1, 2])
    payload = {"schema": AUDIT_SCHEMA, "audit": result.to_dict()}
    payload["run_manifest"] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_sha256": hashlib.sha256(
            (result.reference_report + "\n---\n" + result.candidate_summary).encode("utf-8")
        ).hexdigest(),
        "artifact_role": "expert-quarantined clinical NLI, learned fusion, and ReXErr transfer router",
        **_runtime_provenance(verifier, auditor),
    }
    with left:
        st.download_button(
            "Full audit record (JSON)",
            result_json(payload),
            file_name=f"{EXPORT_PREFIX}_audit_{timestamp}.json",
            mime="application/json",
            use_container_width=True,
        )
    with right:
        st.download_button(
            "Claim-level results (CSV)",
            _rows_csv(_flatten_claim_rows(result, case_id="single-case")),
            file_name=f"{EXPORT_PREFIX}_claims_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with note:
        st.caption("Exports contain supplied report text. Treat them as sensitive research records.")


def _clear_single_audit_state() -> None:
    for key in ("single_audit", "single_audit_ms", "single_audit_input_hash"):
        st.session_state.pop(key, None)


def _load_selected_showcase_case() -> None:
    selected_id = str(
        st.session_state.get("audit_demo_case") or SHOWCASE_CASES[0].case_id
    )
    case = SHOWCASE_CASES_BY_ID.get(selected_id, SHOWCASE_CASES[0])
    st.session_state.audit_demo_case = case.case_id
    st.session_state.audit_reference = case.findings
    st.session_state.audit_candidate = case.impression
    st.session_state.audit_acknowledgement = False
    _clear_single_audit_state()


def _toggle_demo_mode() -> None:
    if st.session_state.get("audit_demo_mode"):
        _load_selected_showcase_case()
        return
    st.session_state.audit_reference = ""
    st.session_state.audit_candidate = ""
    st.session_state.audit_acknowledgement = False
    _clear_single_audit_state()


def audit_workspace() -> None:
    render_shell("Single Report")
    render_notice()
    with st.container(key="mirror_demo_console"):
        demo_mode = st.toggle(
            "Demo Mode",
            key="audit_demo_mode",
            help=(
                "Loads a de-identified teaching case into the complete neural audit. "
                "The result is computed live and is never prefilled."
            ),
            on_change=_toggle_demo_mode,
        )
        if demo_mode:
            selected_id = st.selectbox(
                "Choose a prepared radiology case",
                options=tuple(case.case_id for case in SHOWCASE_CASES),
                format_func=lambda value: SHOWCASE_CASES_BY_ID[value].title,
                key="audit_demo_case",
                on_change=_load_selected_showcase_case,
            )
            selected_case = SHOWCASE_CASES_BY_ID[selected_id]
            st.markdown(
                '<div class="demo-note"><span>Live</span><div>'
                f'<b>{escape(selected_case.title)}.</b> {escape(selected_case.description)} '
                f'<small>Source: {escape(selected_case.origin_report_id)}. '
                'The texts remain editable and run through the real model.</small></div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption(
                "Turn on Demo Mode to load a prepared case, or enter your own "
                "de-identified texts below."
            )
    st.markdown(
        '<div class="step-grid" role="list" aria-label="Audit workflow">'
        '<div class="step" data-step="01" role="listitem"><b>Enter both texts</b><span>Add the source report and summary.</span></div>'
        '<div class="step" data-step="02" role="listitem"><b>Run both directions</b><span>Check additions, conflicts, and missing details.</span></div>'
        '<div class="step" data-step="03" role="listitem"><b>Review the evidence</b><span>Inspect decisions, source traces, and exports.</span></div></div>',
        unsafe_allow_html=True,
    )
    section(
        "Text pair",
        "Add the source report and summary to check",
        "The audit examines what the summary added or changed and what important source content it may have omitted.",
    )
    with st.container(key="mirror_input_grid"):
        left, right = st.columns(2, gap="large")
        with left, st.container(border=True):
            st.markdown(
                '<div class="input-panel-title"><span class="input-number">SRC</span><div>'
                '<strong>Source evidence</strong><span>The report that supports or refutes the summary.</span>'
                '</div></div>',
                unsafe_allow_html=True,
            )
            reference = st.text_area(
                "Source radiology report",
                height=260,
                max_chars=MAX_TEXT_CHARACTERS,
                placeholder="Paste a de-identified Findings section or complete reference report…",
                key="audit_reference",
            )
            st.caption(
                f"{len(reference):,} / {MAX_TEXT_CHARACTERS:,} characters · processed in this app session"
            )
        with right, st.container(border=True):
            st.markdown(
                '<div class="input-panel-title"><span class="input-number">SUM</span><div>'
                '<strong>Candidate summary</strong><span>The generated or edited summary to check.</span>'
                '</div></div>',
                unsafe_allow_html=True,
            )
            candidate = st.text_area(
                "Summary to check",
                height=260,
                max_chars=MAX_TEXT_CHARACTERS,
                placeholder="Paste the de-identified candidate summary or Impression…",
                key="audit_candidate",
            )
            st.caption(
                f"{len(candidate):,} / {MAX_TEXT_CHARACTERS:,} characters · processed in this app session"
            )
    privacy_flags = scan_privacy_risk(reference + "\n" + candidate)
    current_input_hash = hashlib.sha256((reference + "\n---\n" + candidate).encode("utf-8")).hexdigest()
    verifier, auditor = require_runtime()
    missing_text = len(reference.strip()) < 12 or len(candidate.strip()) < 4
    with st.container(key="mirror_action_console"):
        acknowledgement = st.checkbox(
            "I confirm that this text contains no PHI, is de-identified, and is authorized for research processing on this host.",
            key="audit_acknowledgement",
        )
        if privacy_flags:
            categories = ", ".join(
                sorted({flag.category.replace("_", " ") for flag in privacy_flags})
            )
            st.error(
                "Privacy preflight found potential identifiers: "
                f"{categories}. Remove them before running the audit. No matched identifier text is logged."
            )
        run = st.button(
            "Run reciprocal evidence audit",
            type="primary",
            use_container_width=True,
            disabled=auditor is None or not acknowledgement or bool(privacy_flags) or missing_text,
        )
        if missing_text:
            st.caption("Add a meaningful source report and summary to enable the reciprocal audit.")
        elif not acknowledgement:
            st.caption("Confirm de-identification and authorized research use to enable analysis.")
        elif privacy_flags:
            st.caption("Remove the potential identifiers reported by the privacy preflight to continue.")
    if run:
        started = time.perf_counter()
        with st.spinner("Checking added, conflicting, unsupported, and missing report content…"):
            if not INFERENCE_GATE.acquire(timeout=30):
                st.error("The inference worker is busy. Please retry after the active audit finishes.")
                return
            try:
                result = auditor.audit(reference, candidate)
                _assert_neural_active(verifier)
            finally:
                INFERENCE_GATE.release()
        st.session_state.single_audit = result
        st.session_state.single_audit_ms = (time.perf_counter() - started) * 1000
        st.session_state.single_audit_input_hash = current_input_hash
    result = st.session_state.get("single_audit")
    if result and st.session_state.get("single_audit_input_hash") != current_input_hash:
        st.info("The inputs changed after the saved audit. Run the evidence audit again for these texts.")
        result = None
    if result and verifier is not None and auditor is not None:
        st.divider()
        section("Audit outcome", "What needs attention—and why")
        render_audit_result(
            result,
            elapsed_ms=float(st.session_state.get("single_audit_ms", 0.0)),
            review_threshold=float(getattr(verifier, "review_threshold", 0.52)),
            verifier=verifier,
            auditor=auditor,
        )
    else:
        st.markdown(
            '<div class="empty"><b>Your evidence review will appear here.</b><br>'
            'Complete both text fields, confirm de-identification, and select <b>Run reciprocal evidence audit</b>.</div>',
            unsafe_allow_html=True,
        )


def _restore_bidirectional(record: dict[str, Any]) -> BidirectionalAuditResult:
    if record.get("schema") != AUDIT_SCHEMA or not isinstance(record.get("audit"), dict):
        raise ValueError("Unsupported or incomplete saved audit schema.")
    detail = record["audit"]
    return BidirectionalAuditResult(
        reference_report=detail["reference_report"],
        candidate_summary=detail["candidate_summary"],
        commission_audit=AuditResult.from_dict(detail["commission_audit"]),
        omission_audit=AuditResult.from_dict(detail["omission_audit"]),
        commission=DirectionRisk(**detail["commission"]),
        omission=DirectionRisk(**detail["omission"]),
        risk_features=ReportRiskFeatures(**detail["risk_features"]),
        raw_risk=detail["raw_risk"],
        aligned_risk=detail["aligned_risk"],
        alignment_applied=detail["alignment_applied"],
        review_priority=detail["review_priority"],
        scope=detail.get("scope", "textual consistency review"),
    )


def dataset_audit() -> None:
    render_shell("Dataset Audit")
    render_notice()
    st.markdown(
        '<div class="step-grid four" role="list" aria-label="Dataset audit workflow">'
        '<div class="step" data-step="01" role="listitem"><b>Upload</b><span>Add CSV, JSON, or JSONL.</span></div>'
        '<div class="step" data-step="02" role="listitem"><b>Validate</b><span>Check schema, privacy, and rows.</span></div>'
        '<div class="step" data-step="03" role="listitem"><b>Analyze</b><span>Run the neural audit in batches.</span></div>'
        '<div class="step" data-step="04" role="listitem"><b>Review and export</b><span>Prioritize cases and save evidence.</span></div></div>',
        unsafe_allow_html=True,
    )
    section(
        "Dataset input",
        "Upload source-report and summary pairs",
        "Files are validated in session memory, screened for potential identifiers, and processed by the same evidence-grounded neural engine.",
    )
    with st.container(key="mirror_dataset_mode"):
        included_dataset = st.toggle(
            "Use included showcase dataset",
            key="dataset_demo_mode",
            help=(
                "Loads the ready-to-upload teaching dataset into the complete batch workflow. "
                "Every result is computed live by the neural engine."
            ),
        )
        if included_dataset:
            st.markdown(
                '<div class="demo-note"><span>Ready</span><div><b>Curated showcase dataset selected.</b> '
                'It contains de-identified public report excerpts, controlled teaching changes, '
                'and no precomputed model decisions. It is for interface walkthroughs, not a benchmark.'
                '</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption(
                "Upload your own de-identified report pairs, or turn on the included dataset above."
            )
    with st.container(key="mirror_upload_bento"):
        upload_col, template_col = st.columns([2.4, 1])
        with upload_col, st.container(border=True):
            uploaded = st.file_uploader(
                "Report-pair dataset",
                type=("csv", "json", "jsonl"),
                help=(
                    "Required: findings and impression. Optional: case_id, generator, site, split, and subgroup."
                ),
                key="dataset_upload",
                disabled=included_dataset,
            )
            st.caption(
                f"Accepted formats: UTF-8 CSV, JSON, or JSONL · maximum {MAX_UPLOAD_CASES:,} validated pairs"
            )
        with template_col, st.container(border=True):
            st.markdown("**Evidence-pair schema**")
            st.markdown(
                '<div class="schema-list">'
                '<div class="schema-row"><b>findings</b><span>Required source</span></div>'
                '<div class="schema-row"><b>impression</b><span>Required summary</span></div>'
                '<div class="schema-row"><b>case_id</b><span>Optional ID</span></div>'
                '<div class="schema-row"><b>generator · site · split · subgroup</b><span>Optional</span></div>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.download_button(
                "Download ready showcase dataset",
                showcase_dataset_csv(),
                file_name=f"{EXPORT_PREFIX}_showcase_dataset.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "Download empty CSV template",
                upload_template_csv(),
                file_name=f"{EXPORT_PREFIX}_upload_template.csv",
                mime="text/csv",
                use_container_width=True,
            )
    cases: list[dict[str, str]] = []
    payload = b""
    active_filename = ""
    if included_dataset:
        active_filename = f"{EXPORT_PREFIX}_showcase_dataset.csv"
        payload = showcase_dataset_csv().encode("utf-8")
    elif uploaded is not None:
        active_filename = uploaded.name
        payload = uploaded.getvalue()
    if payload:
        try:
            cases = parse_uploaded_cases(active_filename, payload)
        except UploadValidationError as exc:
            st.error(str(exc))
        else:
            file_hash = hashlib.sha256(payload).hexdigest()
            columns = st.columns(4)
            columns[0].metric("Valid cases", f"{len(cases):,}")
            columns[1].metric("File size", f"{len(payload) / 1024:.1f} KB")
            columns[2].metric("Schema", "Validated")
            columns[3].metric("SHA-256", file_hash[:12] + "…")
            with st.expander("Preflight details"):
                st.json(
                    {
                        "filename": active_filename,
                        "sha256": file_hash,
                        "encoding": "UTF-8",
                        "rows": len(cases),
                        "mapped_fields": sorted({key for case in cases for key in case}),
                        "duplicate_case_ids": 0,
                        "data_retention": "session memory only",
                    }
                )
    acknowledgement = st.checkbox(
        "I confirm this dataset contains no PHI, is de-identified, and is authorized for research processing on this host.",
        key="batch_acknowledgement",
    )
    privacy_findings = {
        case["case_id"]: scan_privacy_risk(case["findings"] + "\n" + case["impression"])
        for case in cases
    }
    flagged_cases = {case_id: flags for case_id, flags in privacy_findings.items() if flags}
    if flagged_cases:
        categories = sorted(
            {
                flag.category.replace("_", " ")
                for flags in flagged_cases.values()
                for flag in flags
            }
        )
        st.error(
            f"Privacy preflight blocked {len(flagged_cases):,} case(s). Potential identifier categories: "
            + ", ".join(categories)
            + ". Remove them and upload the corrected file."
        )
    verifier, auditor = require_runtime()
    run = st.button(
        f"Run reciprocal audit on {len(cases):,} report pairs" if cases else "Run dataset audit",
        type="primary",
        use_container_width=True,
        disabled=not cases or not acknowledgement or auditor is None or bool(flagged_cases),
    )
    if run and cases:
        results: list[dict[str, Any]] = []
        detailed: list[dict[str, Any]] = []
        claim_rows: list[dict[str, Any]] = []
        batch_size = BATCH_REPORTS
        progress = st.progress(0.0, text="Preparing neural batch…")
        started = time.perf_counter()

        def record_success(case: dict[str, str], audit: BidirectionalAuditResult) -> None:
            metadata = {
                key: case[key]
                for key in ("generator", "site", "split", "subgroup")
                if key in case
            }
            results.append(
                {
                    "case_id": case["case_id"],
                    **metadata,
                    "status": "completed",
                    "review_priority": audit.review_priority,
                    "aligned_risk": round(audit.aligned_risk, 6),
                    "commission_risk": round(audit.commission.score, 6),
                    "omission_risk": round(audit.omission.score, 6),
                    "commission_alerts": audit.commission.alerted_claims,
                    "omission_alerts": audit.omission.alerted_claims,
                    "error_types": "; ".join(
                        sorted(
                            {
                                value.value
                                for direction in (audit.commission_audit, audit.omission_audit)
                                for claim in direction.claims
                                for value in claim.error_types
                                if value.value != "NONE"
                            }
                        )
                    ),
                    "error": "",
                }
            )
            detailed.append(
                {
                    "schema": AUDIT_SCHEMA,
                    "case_id": case["case_id"],
                    "metadata": metadata,
                    "audit": audit.to_dict(),
                }
            )
            claim_rows.extend(
                {**metadata, **row}
                for row in _flatten_claim_rows(audit, case_id=case["case_id"])
            )

        def record_failure(case: dict[str, str], exc: Exception) -> None:
            metadata = {
                key: case[key]
                for key in ("generator", "site", "split", "subgroup")
                if key in case
            }
            results.append(
                {
                    "case_id": case["case_id"],
                    **metadata,
                    "status": "failed",
                    "review_priority": "",
                    "aligned_risk": "",
                    "commission_risk": "",
                    "omission_risk": "",
                    "commission_alerts": 0,
                    "omission_alerts": 0,
                    "error_types": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

        for start in range(0, len(cases), batch_size):
            batch = cases[start : start + batch_size]
            try:
                if not INFERENCE_GATE.acquire(timeout=30):
                    raise RuntimeError("Inference worker remained busy for 30 seconds.")
                try:
                    audits = auditor.audit_many(
                        [(row["findings"], row["impression"]) for row in batch],
                        nli_batch_size=NLI_BATCH_SIZE,
                    )
                    _assert_neural_active(verifier)
                finally:
                    INFERENCE_GATE.release()
                for case, audit in zip(batch, audits, strict=True):
                    record_success(case, audit)
            except Exception:  # noqa: BLE001 - isolate and retry each case in a failed chunk
                for case in batch:
                    try:
                        if not INFERENCE_GATE.acquire(timeout=30):
                            raise RuntimeError("Inference worker remained busy for 30 seconds.")
                        try:
                            audit = auditor.audit(case["findings"], case["impression"])
                            _assert_neural_active(verifier)
                        finally:
                            INFERENCE_GATE.release()
                    except Exception as case_exc:  # noqa: BLE001 - record per-case failure
                        record_failure(case, case_exc)
                    else:
                        record_success(case, audit)
            completed = min(start + len(batch), len(cases))
            progress.progress(completed / len(cases), text=f"Audited {completed:,} of {len(cases):,} cases")
        progress.empty()
        st.session_state.batch_run = {
            "schema": BATCH_SCHEMA,
            "manifest": {
                "generated_at_utc": datetime.now(UTC).isoformat(),
                "input_sha256": hashlib.sha256(payload).hexdigest(),
                "input_cases": len(cases),
                "elapsed_seconds": time.perf_counter() - started,
                **_runtime_provenance(verifier, auditor),
            },
            "summary": results,
            "claims": claim_rows,
            "details": detailed,
        }
    batch_run = st.session_state.get("batch_run")
    current_upload_hash = hashlib.sha256(payload).hexdigest() if payload else None
    if batch_run and batch_run.get("manifest", {}).get("input_sha256") != current_upload_hash:
        st.info("The uploaded dataset changed after the saved batch run. Audit the current file again.")
        batch_run = None
    if batch_run:
        if batch_run.get("schema") != BATCH_SCHEMA:
            st.error("The saved batch result uses an incompatible schema and cannot be displayed.")
            return
        st.divider()
        section("Dataset outcome", "Prioritized review queue")
        rows = batch_run["summary"]
        complete = [row for row in rows if row["status"] == "completed"]
        columns = st.columns(4)
        columns[0].metric("Completed", f"{len(complete):,}")
        columns[1].metric("High priority", sum(row["review_priority"] == "HIGH" for row in complete))
        columns[2].metric("Review", sum(row["review_priority"] == "REVIEW" for row in complete))
        columns[3].metric("Failed", f"{len(rows) - len(complete):,}")
        filter_columns = st.columns(2)
        priorities = filter_columns[0].multiselect(
            "Filter review priority",
            options=("HIGH", "REVIEW", "ROUTINE"),
            default=("HIGH", "REVIEW", "ROUTINE"),
        )
        error_options = sorted(
            {
                item
                for row in complete
                for item in str(row.get("error_types", "")).split("; ")
                if item
            }
        )
        selected_errors = filter_columns[1].multiselect(
            "Filter error type",
            options=error_options,
        )
        displayed = [
            row
            for row in rows
            if row["status"] != "completed"
            or (
                row["review_priority"] in priorities
                and (
                    not selected_errors
                    or any(item in str(row.get("error_types", "")) for item in selected_errors)
                )
            )
        ]
        displayed.sort(
            key=lambda row: (
                row["status"] != "completed",
                -float(row["aligned_risk"]) if row["status"] == "completed" else 0.0,
            )
        )
        display_rows = []
        for row in displayed:
            completed = row.get("status") == "completed"
            display_rows.append(
                {
                    "Case": row.get("case_id"),
                    "Status": str(row.get("status", "")).title(),
                    "Review priority": row.get("review_priority") or "—",
                    "Overall review score": (
                        f"{float(row['aligned_risk']):.1%}" if completed else "—"
                    ),
                    "Added/conflicting score": (
                        f"{float(row['commission_risk']):.1%}" if completed else "—"
                    ),
                    "Missing-content score": (
                        f"{float(row['omission_risk']):.1%}" if completed else "—"
                    ),
                    "Flagged claim types": row.get("error_types") or "—",
                    "Generator": row.get("generator") or "—",
                    "Site": row.get("site") or "—",
                    "Error": row.get("error") or "",
                }
            )
        st.dataframe(display_rows, hide_index=True, width="stretch")
        if complete:
            import plotly.express as px

            figure = px.scatter(
                complete,
                x="commission_risk",
                y="omission_risk",
                color="review_priority",
                symbol="review_priority",
                hover_name="case_id",
                color_discrete_map={"ROUTINE": "#16734f", "REVIEW": "#9a6200", "HIGH": "#a62b22"},
                labels={
                    "commission_risk": "Added or conflicting content score",
                    "omission_risk": "Missing-content score",
                    "review_priority": "Review priority",
                },
                title="Added or conflicting content versus missing content",
            )
            figure.update_layout(
                height=410,
                legend_title_text="Priority",
                margin={"l": 10, "r": 10, "t": 55, "b": 10},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#f8fafb",
                font={"family": "Inter, Aptos, Segoe UI, sans-serif", "color": "#344c5e"},
            )
            figure.update_traces(marker={"size": 10, "line": {"width": 1, "color": "#ffffff"}})
            st.plotly_chart(figure, width="stretch")
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        columns = st.columns(4)
        columns[0].download_button(
            "Review queue (CSV)",
            _rows_csv(rows),
            file_name=f"{EXPORT_PREFIX}_batch_summary_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        columns[1].download_button(
            "Claim evidence (CSV)",
            _rows_csv(batch_run["claims"]),
            file_name=f"{EXPORT_PREFIX}_batch_claims_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        columns[2].download_button(
            "Reproducibility record (JSON)",
            result_json(batch_run),
            file_name=f"{EXPORT_PREFIX}_batch_assurance_{timestamp}.json",
            mime="application/json",
            use_container_width=True,
        )
        failures = [row for row in rows if row["status"] == "failed"]
        columns[3].download_button(
            "Processing failures (CSV)",
            _rows_csv(failures),
            file_name=f"{EXPORT_PREFIX}_batch_failures_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=not failures,
        )
        if batch_run["details"]:
            selected = st.selectbox(
                "Open a completed report pair",
                options=range(len(batch_run["details"])),
                format_func=lambda index: batch_run["details"][index]["case_id"],
            )
            render_audit_result(
                _restore_bidirectional(batch_run["details"][selected]),
                elapsed_ms=0.0,
                review_threshold=float(verifier.review_threshold),
                verifier=verifier,
                auditor=auditor,
            )


def _metric_value(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _ci(summary: dict[str, Any], name: str) -> str:
    block = summary.get("confidence_intervals", {}).get(name, {})
    if block.get("lower") is None:
        return "95% CI unavailable"
    return f"95% CI {float(block['lower']):.3f}–{float(block['upper']):.3f}"


@st.cache_data(show_spinner=False)
def _cached_file_hash(path: str, size: int, modified_ns: int) -> str:
    del size, modified_ns
    return _sha256(Path(path))


def _verified_prediction_ledgers(summary: dict[str, Any]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    for block in (
        summary["rexerr"]["files"],
        summary["rexerr"]["report_level"]["files"],
        summary["radevalx"]["files"],
    ):
        for key in ("predictions", "identity_predictions"):
            configured = block.get(key)
            if not configured:
                continue
            expected = block.get(f"{key}_sha256")
            path = Path(str(configured))
            path = path if path.is_absolute() else PROJECT_ROOT / path
            if not path.is_file() or not expected:
                issues.append(f"{key}: missing file or checksum")
                continue
            stat = path.stat()
            actual = _cached_file_hash(str(path.resolve()), stat.st_size, stat.st_mtime_ns)
            if actual != expected:
                issues.append(f"{key}: checksum mismatch")
    return not issues, issues


def _evaluation_ledger() -> dict[str, Any] | None:
    summary = _read_json(EVALUATION_SUMMARY)
    required = (
        summary.get("schema") == "mirror-rad.evaluation.v1",
        summary.get("project") == PRODUCT_NAME,
        summary.get("completion_status") == "complete",
        isinstance(summary.get("controlled", {}).get("variants"), dict),
        isinstance(summary.get("adaptation_holdout", {}).get("variants"), dict),
        isinstance(summary.get("rexerr", {}).get("sentence_level", {}).get("metrics"), dict),
        isinstance(summary.get("rexerr", {}).get("report_level", {}).get("metrics"), dict),
        isinstance(summary.get("radevalx", {}).get("metrics"), dict),
        "count_aware_bidirectional_raw"
        in summary.get("radevalx", {}).get("model_ablations", {}),
    )
    if not all(required):
        st.error(
            "The evaluation ledger is missing, incomplete, or incompatible with this release. "
            "Measured values are hidden rather than replaced with defaults."
        )
        return None
    verified, issues = _verified_prediction_ledgers(summary)
    if not verified:
        st.error("One or more saved prediction ledgers failed integrity validation: " + "; ".join(issues))
        return None
    return summary


def evaluation_hub() -> None:
    render_shell("Results & Benchmarks")
    section(
        "Evidence overview",
        "What each benchmark tells us",
        "Claim classification, external transfer, human-consensus ranking, and review efficiency answer different questions and are reported separately.",
    )
    summary = _evaluation_ledger()
    if summary is None:
        return
    headline = summary.get("headline", {})
    controlled_variants = summary["controlled"]["variants"]
    controlled_primary = controlled_variants["learned_with_rules_calibrated_no_override"]
    st.markdown(
        '<div class="benchmark-grid" role="list" aria-label="Benchmark overview">'
        '<article class="benchmark-card" role="listitem"><div class="question">Can it classify claims?</div>'
        '<h3>Controlled claim classification</h3>'
        f'<strong>{_metric_value(controlled_primary["classification"]["macro_f1"])}</strong>'
        '<p>Macro-F1 on the expert-family-quarantined controlled test.</p></article>'
        '<article class="benchmark-card" role="listitem"><div class="question">Does it transfer?</div>'
        '<h3>External report ranking</h3>'
        f'<strong>{_metric_value(headline["rexerr_report_auroc"])}</strong>'
        '<p>AUROC on patient-disjoint changed ReXErr reports.</p></article>'
        '<article class="benchmark-card" role="listitem"><div class="question">Does it align with reviewers?</div>'
        '<h3>Human-consensus ranking</h3>'
        f'<strong>{_metric_value(headline["radevalx_auroc"])}</strong>'
        '<p>AUROC on 100 genuine generated reports reviewed by two radiologists.</p></article>'
        '<article class="benchmark-card" role="listitem"><div class="question">Does triage help?</div>'
        '<h3>Error burden found at 20%</h3>'
        f'<strong>{float(headline["radevalx_significant_burden_capture_at_20pct"]):.1%}</strong>'
        '<p>Significant consensus error burden captured when reviewing one fifth of reports.</p></article></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Every value is loaded from the sealed evaluation ledger after its prediction files pass SHA-256 verification. "
        "No reviewer labels were generated by this project."
    )
    expert_tab, report_tab, sentence_tab, adaptation_tab, controlled_tab, integrity_tab = st.tabs(
        [
            "Human consensus",
            "External transfer · reports",
            "External transfer · sentences",
            "Sealed adaptation holdout",
            "Controlled ablations",
            "Validity & integrity",
        ]
    )
    with expert_tab:
        data = summary["radevalx"]
        population = data["population"]
        st.markdown("### RadEvalX · two-radiologist consensus")
        st.caption(
            f"{population['reports']} genuine generated reports · "
            f"{population['significant_positive_reports']} with significant errors · "
            f"{population['significant_error_count']} significant errors and "
            f"{population['insignificant_error_count']} insignificant errors by consensus."
        )
        metrics = data["metrics"]
        columns = st.columns(4)
        columns[0].metric("AUROC", _metric_value(metrics["auroc"]), help=_ci(data, "auroc"))
        columns[1].metric("AUPRC", _metric_value(metrics["auprc"]), help=_ci(data, "auprc"))
        columns[2].metric(
            "Significant-burden Spearman",
            _metric_value(data["significant_burden_correlation"]["spearman"]),
        )
        columns[3].metric(
            "Burden captured @20%",
            f"{float(data['review_budgets']['20pct']['error_count_recall']):.1%}",
        )
        st.dataframe(
            [
                {
                    "Review budget": label,
                    "Reports reviewed": value.get("reviewed"),
                    "Positive-report recall": value.get("report_recall"),
                    "Significant burden captured": value.get("error_count_recall"),
                    "Queue precision": value.get("precision"),
                }
                for label, value in data["review_budgets"].items()
            ],
            hide_index=True,
            width="stretch",
        )
        rows = []
        for key, label in (
            ("count_aware_bidirectional_raw", f"{PRODUCT_NAME} count-aware bidirectional"),
            ("legacy_max_top3_bidirectional_raw", f"{PRODUCT_NAME} legacy max/top-3"),
            ("rexerr_transfer_router", f"{PRODUCT_NAME} ReXErr transfer router"),
        ):
            model = data["model_ablations"][key]
            rows.append(
                {
                    "System": label,
                    "AUROC": model["metrics"]["auroc"],
                    "AUPRC": model["metrics"]["auprc"],
                    "Burden captured @20%": model["review_budgets"]["20pct"]["error_count_recall"],
                }
            )
        for name, value in data["established_baselines"].items():
            rows.append(
                {
                    "System": name,
                    "AUROC": value.get("metrics", {}).get("auroc"),
                    "AUPRC": value.get("metrics", {}).get("auprc"),
                    "Burden captured @20%": value.get("review_budgets", {}).get("20pct", {}).get("error_count_recall"),
                }
            )
        st.markdown("#### Established metric comparison")
        st.dataframe(rows, hide_index=True, width="stretch")
        chexbert = data["established_baselines"].get("CheXbert", {})
        comparison = chexbert.get("paired_difference_vs_mirror", {})
        auroc_difference = comparison.get("auroc", {})
        lower = auroc_difference.get("lower")
        upper = auroc_difference.get("upper")
        if lower is not None and upper is not None and lower <= 0.0 <= upper:
            conclusion = "The paired interval crosses zero, so neither system has a statistically resolved AUROC lead."
        elif lower is not None and lower > 0.0:
            conclusion = f"{PRODUCT_NAME} has a positive paired AUROC difference on this cohort."
        else:
            conclusion = "CheXbert has a negative paired AUROC difference on this cohort."
        st.info(
            f"CheXbert AUROC is {_metric_value(chexbert.get('metrics', {}).get('auroc'))}; "
            f"{PRODUCT_NAME} is {_metric_value(metrics.get('auroc'))}. "
            f"Paired difference ({PRODUCT_NAME} − CheXbert): "
            f"{_metric_value(auroc_difference.get('estimate'))} "
            f"(95% CI {_metric_value(lower)} to {_metric_value(upper)}). {conclusion} "
            "This is an iterative 100-report external diagnostic, not clinical-effectiveness evidence."
        )
    with report_tab:
        data = summary["rexerr"]["report_level"]
        population = data["population"]
        st.markdown("### ReXErr · changed-report transfer routing")
        st.caption(
            f"{population['cases']:,} changed reports from {population['unique_subjects']:,} patients · "
            f"{population['error_reports']:,} semantic-positive · "
            f"{population['changed_semantic_negative']:,} linguistic-only changed controls."
        )
        metrics = data["metrics"]
        columns = st.columns(4)
        columns[0].metric("AUROC", _metric_value(metrics["auroc"]), help=_ci(data, "auroc"))
        columns[1].metric("AUPRC", _metric_value(metrics["auprc"]), help=_ci(data, "auprc"))
        columns[2].metric("Recall", f"{float(metrics['recall']):.1%}", help=_ci(data, "recall"))
        columns[3].metric("Specificity", f"{float(metrics['specificity']):.1%}", help=_ci(data, "specificity"))
        controls = data["baselines_and_negative_controls"]
        control_columns = st.columns(3)
        control_columns[0].metric(
            "Changed-control specificity",
            f"{float(controls['changed_semantic_negative_slice']['specificity']):.1%}",
        )
        control_columns[1].metric(
            "Identity false-alert rate",
            f"{float(controls['identity_invariance']['false_alert_rate']):.1%}",
        )
        control_columns[2].metric("Report-score ECE", _metric_value(metrics["ece"]))
        router = data["transfer_router"]
        st.dataframe(
            [
                {
                    "Router property": "Method",
                    "Value": str(router["selected_method"]),
                },
                {
                    "Router property": "Fit rows",
                    "Value": f"{int(router['fit_population']['cases']):,}",
                },
                {
                    "Router property": "Validation rows",
                    "Value": f"{int(router['selection_population']['cases']):,}",
                },
                {
                    "Router property": "Review threshold",
                    "Value": _metric_value(router["review_threshold"], 4),
                },
                {
                    "Router property": "High-priority threshold",
                    "Value": _metric_value(router["high_priority_threshold"], 4),
                },
            ],
            hide_index=True,
            width="stretch",
        )
        identity_false_alert = float(
            controls["identity_invariance"]["false_alert_rate"]
        )
        st.warning(
            f"This population is extremely imbalanced and has only "
            f"{population['changed_semantic_negative']:,} changed semantic-negative reports. "
            f"The separate identity-control false-alert rate is {identity_false_alert:.1%}. "
            "The output is an ordinal review-prioritization score, not a radiologist probability."
        )
    with sentence_tab:
        data = summary["rexerr"]
        metrics = data["sentence_level"]["metrics"]
        population = data["primary_population"]
        st.markdown("### ReXErr · contextual sentence diagnostic")
        st.caption(
            f"{population['rows']:,} contextual sentence cases · {population['semantic_errors']:,} semantic errors · "
            f"{population['clean_controls']:,} clean controls. Each sentence is checked against its complete paired report."
        )
        columns = st.columns(4)
        columns[0].metric("AUROC", _metric_value(metrics["auroc"]), help=_ci(data, "auroc"))
        columns[1].metric("AUPRC", _metric_value(metrics["auprc"]), help=_ci(data, "auprc"))
        columns[2].metric("Recall", f"{float(metrics['recall']):.1%}", help=_ci(data, "recall"))
        columns[3].metric("Specificity", f"{float(metrics['specificity']):.1%}", help=_ci(data, "specificity"))
        st.dataframe(
            [
                {
                    "Error type": name,
                    "N": value["n"],
                    "Recall": value["recall"],
                    "Mean score": value["mean_risk"],
                }
                for name, value in data["error_type_recall"].items()
            ],
            hide_index=True,
            width="stretch",
        )
        hardest = sorted(
            data["error_type_recall"].items(),
            key=lambda item: float(item[1].get("recall", 0.0)),
        )[:4]
        hard_text = "; ".join(
            f"{name}: {float(value['recall']):.1%}"
            for name, value in hardest
        )
        strict = data.get("strict_decontaminated", {})
        strict_population = strict.get("population", {})
        strict_metrics = strict.get("metrics", {})
        st.info(
            f"Hardest current semantic slices: {hard_text}. "
            "Typos, homophones, and repetition remain separate linguistic controls."
        )
        if strict_population:
            st.markdown("#### Strict unseen-pair audit")
            strict_columns = st.columns(4)
            strict_columns[0].metric("Rows", f"{int(strict_population['rows']):,}")
            strict_columns[1].metric("AUROC", _metric_value(strict_metrics.get("auroc")))
            strict_columns[2].metric("AUPRC", _metric_value(strict_metrics.get("auprc")))
            strict_columns[3].metric("Recall", f"{float(strict_metrics.get('recall', 0.0)):.1%}")
            st.caption(
                "This stricter slice removes exact normalized mutation pairs seen in official ReXErr train or validation."
            )
    with adaptation_tab:
        holdout = summary["adaptation_holdout"]
        holdout_variants = holdout["variants"]
        holdout_primary = holdout_variants["learned_with_rules_calibrated_no_override"]
        holdout_classification = holdout_primary["classification"]
        holdout_classes = holdout_classification["per_class"]
        st.markdown("### Patient-disjoint adaptation holdout")
        st.caption(
            f"{int(holdout['evaluation_rows']):,} claims from the ReXErr-training adaptation pool were kept "
            "behind a subject and exact-pair boundary. The original single-open result was retained before a "
            "subsequent multi-domain fusion correction; the values below are the transparent final-artifact "
            "rerun and are therefore iterative evidence."
        )
        columns = st.columns(4)
        columns[0].metric("Macro-F1", _metric_value(holdout_classification.get("macro_f1"), 4))
        columns[1].metric(
            "Contradiction recall",
            f"{float(holdout_classes['CONTRADICTED']['recall']):.1%}",
        )
        columns[2].metric(
            "Unsupported recall",
            f"{float(holdout_classes['UNSUPPORTED']['recall']):.1%}",
        )
        columns[3].metric("ECE", _metric_value(holdout_primary["calibration"].get("ece"), 4))
        slice_rows = []
        for name, value in holdout_primary.get("slices", {}).items():
            slice_rows.append(
                {
                    "Clinical slice": name.replace("_", " ").title(),
                    "N": value.get("n"),
                    "Alert recall": value.get("alert_recall"),
                    "Contradiction recall": value.get("contradiction_recall"),
                    "Accuracy": value.get("accuracy"),
                }
            )
        if slice_rows:
            st.dataframe(slice_rows, hide_index=True, width="stretch")
        st.info(
            "The original frozen result remains in the evidence ledger, but this final-artifact rerun is not "
            "presented as a new pristine holdout. ReXErr official test and RadEvalX are likewise external "
            "iterative diagnostics because earlier development inspected them."
        )
    with controlled_tab:
        variants = controlled_variants
        st.dataframe(
            [
                {
                    "Variant": name.replace("_", " ").title(),
                    "Macro-F1": value["classification"]["macro_f1"],
                    "Contradiction recall": value["classification"]["per_class"]["CONTRADICTED"]["recall"],
                    "Unsupported recall": value["classification"]["per_class"]["UNSUPPORTED"]["recall"],
                    "ECE": value["calibration"]["ece"],
                }
                for name, value in variants.items()
            ],
            hide_index=True,
            width="stretch",
        )
        primary_classification = controlled_primary["classification"]
        safety_variant = variants.get("full_system_no_abstention", controlled_primary)
        st.caption(
            f"Primary controlled classifier: macro-F1 "
            f"{_metric_value(primary_classification['macro_f1'], 4)}, contradiction recall "
            f"{_metric_value(primary_classification['per_class']['CONTRADICTED']['recall'], 4)}, "
            f"unsupported recall {_metric_value(primary_classification['per_class']['UNSUPPORTED']['recall'], 4)}, "
            f"ECE {_metric_value(controlled_primary['calibration']['ece'], 4)}. "
            f"The separately labelled safety configuration has contradiction recall "
            f"{_metric_value(safety_variant['classification']['per_class']['CONTRADICTED']['recall'], 4)}. "
            "These are controlled claim-level results—not overall system accuracy."
        )
    with integrity_tab:
        integrity = summary["integrity"]
        quarantine = integrity["expert_quarantine"]
        columns = st.columns(3)
        columns[0].metric("Expert reports quarantined", quarantine["expert_reports_quarantined"])
        columns[1].metric("Content groups quarantined", quarantine["content_groups_quarantined"])
        columns[2].metric("Neural fallback", "Never used" if not integrity["neural_fallback_used"] else "Detected")
        if quarantine.get("status") == "passed" and not integrity["neural_fallback_used"]:
            st.success(
                "Expert-family quarantine passed and the frozen evaluation completed without neural fallback."
            )
        else:
            st.error("The release failed one or more integrity gates.")
        release = artifact_health()["release"]
        st.code(f"Sealed release ID: {release.get('release_id')}")
        st.markdown("#### Evidence boundaries")
        for item in summary["radevalx"]["limitations"]:
            st.markdown(f"- {item}")
        for item in summary["rexerr"].get("limitations", []):
            st.markdown(f"- {item}")
        st.warning(summary["claim_boundary"])
        st.info(
            "The retained generated-summary corpus is unlabelled and unscored. No accuracy, kappa, or factuality "
            "claim is made for it without independent annotations."
        )


def research_ledger() -> None:
    render_shell("Study Design")
    section("Research problem", "What the project is designed to solve")
    st.markdown(
        '<div class="research-summary" role="list">'
        '<article class="research-card" role="listitem"><span>Problem</span><h3>Summary errors are hard to localize</h3>'
        '<p>Generated radiology summaries can add unsupported findings, reverse clinical attributes, or omit important source statements. Word overlap does not explain these mechanisms.</p></article>'
        '<article class="research-card" role="listitem"><span>Objective</span><h3>Build an evidence-grounded auditor</h3>'
        '<p>Detect additions, conflicts, unsupported claims, and omissions; expose the source path; abstain under uncertainty; and produce a reproducible review priority.</p></article>'
        '<article class="research-card" role="listitem"><span>Primary question</span><h3>Does bidirectional evidence improve review ranking?</h3>'
        '<p>Measure how reliably the leakage-quarantined system ranks radiologist-identified significant errors compared with established automatic report metrics.</p></article></div>',
        unsafe_allow_html=True,
    )
    section(
        "Mutual-inference architecture",
        "Five traceable layers working in both directions",
        "Forward retrieval tests what the summary added; reciprocal retrieval tests what the summary failed to preserve.",
    )
    st.markdown(
        '<div class="flow" role="list" aria-label="Evidence audit architecture">'
        '<div class="flow-node" role="listitem"><b>Atomic claims</b><span>Split reports into checkable statements.</span></div>'
        '<div class="flow-node" role="listitem"><b>Dual evidence retrieval</b><span>Combine raw and clinical-slot-masked matching in both directions.</span></div>'
        '<div class="flow-node" role="listitem"><b>Adapted clinical NLI</b><span>Classify support, conflict, or missing support in full report context.</span></div>'
        '<div class="flow-node" role="listitem"><b>Semantic delta</b><span>Bind device identity, position, anatomy, severity, and time to entities.</span></div>'
        '<div class="flow-node" role="listitem"><b>Burden-aware routing</b><span>Preserve both single critical errors and cumulative omission burden.</span></div></div>',
        unsafe_allow_html=True,
    )
    columns = st.columns(3)
    cards = [
        ("Human consensus", "RadEvalX", "100 genuine generated reports with significant and insignificant consensus error counts from two board-certified radiologists."),
        ("External silver", "ReXErr", "Patient-disjoint MIMIC-CXR context evaluation and report-router adaptation, with linguistic-only changes kept separate."),
        ("Controlled", "Open-I", "Three-way claim NLI and error-type ablations after quarantining all expert-linked report families."),
    ]
    for column, (badge, title, copy) in zip(columns, cards):
        with column:
            st.markdown(
                f'<article class="dataset-card"><span class="badge">{escape(badge)}</span><h3>{escape(title)}</h3><p>{escape(copy)}</p></article>',
                unsafe_allow_html=True,
            )
    section("Innovation", "Why this is more than an NLI classifier")
    st.markdown(
        "- **Bidirectional evidence graph:** forward edges detect additions and contradictions; reverse edges expose missing reference content.\n"
        "- **Slot-masked reciprocal retrieval:** a changed device, position, severity, or location cannot hide its true evidence counterpart through lexical drift alone.\n"
        "- **Entity-bound semantic deltas:** device identity and position, anatomy, severity, and temporal change are compared locally with synonym-safe clinical ontologies.\n"
        "- **Patient-disjoint adaptation:** continued NLI training, fusion fitting, calibration, and a sealed holdout occupy separate SHA-256 subject partitions.\n"
        "- **Leakage quarantine by content family:** all 97 expert-linked duplicate groups are removed before neural training, fusion training, selection, and controlled testing.\n"
        "- **Count-aware review routing:** ranking retains cumulative claim error mass and noisy-OR evidence instead of discarding everything below the top three claims.\n"
        "- **Review-budget evaluation:** success is measured by how much expert error burden is captured when only 10%, 20%, or 30% of reports can be reviewed.\n"
        "- **Decision trace:** every alert retains its atomic claim, evidence, probabilities, consistency checks, direction, and confidence."
    )
    section("Responsible use", "What the outputs mean—and do not mean")
    st.warning(RESEARCH_NOTICE)
    st.markdown(
        "A **Supported** label means the supplied reference text appears to support the claim. It does not mean the statement is correct on the image. A **review risk** is a queue-prioritization signal, not probability of disease, harm, or diagnostic error."
    )


def system_status() -> None:
    render_shell("System Readiness")
    section("Readiness check", "Verified model, evidence, and release assets")
    health = artifact_health()
    ready_title = "Ready for research auditing" if health["ready"] else "Attention required before auditing"
    ready_copy = (
        "All required model, evidence, and integrity gates passed."
        if health["ready"]
        else "One or more required assets failed validation; inference will fail closed."
    )
    state_class = "readiness-hero" if health["ready"] else "readiness-hero attention"
    badge_text = "All gates passed" if health["ready"] else "Action required"
    st.markdown(
        f'<section class="{state_class}" aria-label="System readiness"><div><h3>{escape(ready_title)}</h3>'
        f'<p>{escape(ready_copy)}</p></div><span class="readiness-badge">{escape(badge_text)}</span></section>',
        unsafe_allow_html=True,
    )
    component_labels = {
        "verifier_configuration": "Verifier configuration",
        "fusion_model": "Learned evidence fusion",
        "clinical_nli_checkpoint": "Clinical NLI model",
        "complete_report_router": "Complete-report review router",
        "evaluation_ledger": "Evaluation evidence ledger",
        "sealed_release_manifest": "Sealed release integrity",
    }
    check_rows = []
    for name, ready in health["checks"].items():
        status_class = "check-status" if ready else "check-status missing"
        icon = "✓" if ready else "!"
        text_value = "Ready" if ready else "Missing"
        check_rows.append(
            f'<div class="check-row" role="listitem"><span>{escape(component_labels.get(name, name))}</span>'
            f'<span class="{status_class}"><span class="check-icon" aria-hidden="true">{icon}</span>{text_value}</span></div>'
        )
    st.markdown(
        '<div class="check-list" role="list" aria-label="Release readiness checks">'
        + "".join(check_rows)
        + "</div>",
        unsafe_allow_html=True,
    )
    if health["ready"]:
        st.success(
            "The neural checkpoint, learned claim fusion, complete-report router, evaluation ledger, "
            "application code, and dependency lock match the sealed release manifest."
        )
    else:
        st.error(
            "One or more release gates failed. The application fails closed rather than substituting a weaker model."
        )
        errors = health.get("release", {}).get("errors", [])
        if errors:
            with st.expander("Release validation failures"):
                for error in errors:
                    st.markdown(f"- {error}")
    config = health.get("config", {})
    router = health.get("calibrator_metadata", {})
    columns = st.columns(4)
    columns[0].metric("Evidence retrieval", f"Top-{config.get('top_k', '—')}")
    columns[1].metric("Review threshold", _metric_value(router.get("review_threshold"), 2))
    columns[2].metric("Clinical NLI", "Configured" if config.get("nli_model") else "Missing")
    build_id = str(health.get("release", {}).get("release_id") or "Unsealed")[:12]
    columns[3].metric("Build ID", build_id)
    st.caption(
        f"Build {build_id} is a software-integrity fingerprint showing that the tested code, model, "
        "configuration, and evidence files still match. It is not an accuracy score, clinical "
        "approval, or certification."
    )
    with st.expander("Advanced diagnostics · artifact integrity hashes"):
        hashes = {}
        for name in ("verifier_config.json", "fusion_model.pkl"):
            path = health["artifact"] / name
            hashes[name] = _sha256(path) if path.is_file() else None
        hashes["risk_calibrator"] = _sha256(health["calibrator"]) if health["calibrator"].is_file() else None
        hashes["evaluation_ledger"] = _sha256(EVALUATION_SUMMARY) if EVALUATION_SUMMARY.is_file() else None
        hashes["release_manifest"] = _sha256(RELEASE_MANIFEST) if RELEASE_MANIFEST.is_file() else None
        st.json(hashes)
    if st.button("Run system check", type="primary"):
        verifier, auditor = require_runtime()
        if auditor is not None:
            with st.spinner("Executing a real bidirectional inference self-check…"):
                if not INFERENCE_GATE.acquire(timeout=30):
                    st.error("The inference worker is currently busy.")
                    return
                try:
                    result = auditor.audit(
                        "No focal airspace opacity, pleural effusion, or pneumothorax.",
                        "No acute cardiopulmonary abnormality.",
                    )
                    _assert_neural_active(verifier)
                finally:
                    INFERENCE_GATE.release()
            st.success(
                f"Neural self-check passed. Both directions completed with transfer score "
                f"{result.aligned_risk:.3f}."
            )
    section("Privacy and deployment", "Session-only research service")
    st.markdown(
        "The application uses bundled offline model artifacts, session-memory uploads, UTF-8 validation, bounded inputs, sanitized spreadsheet exports, portable model paths, and a fail-closed neural runtime. Online hosting receives input over the network before the application can screen it. Public hosting is for synthetic or de-identified research text only and must never receive PHI or identifiable patient data."
    )


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    page = render_navigation()
    if page == "Single Report":
        audit_workspace()
    elif page == "Dataset Audit":
        dataset_audit()
    elif page == "Results & Benchmarks":
        evaluation_hub()
    elif page == "Study Design":
        research_ledger()
    else:
        system_status()


if __name__ == "__main__":
    main()
