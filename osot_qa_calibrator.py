#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OSOT 标注质检、双人盲标一致性 (IAA) 与规则自适应校准器 (osot_qa_calibrator.py)
==========================================================================
专为 OSOT 下一周期任务 T2（JSON 质检与阈值调优）及论文双人盲标实验打造的原创工具。

四大核心功能（纯 Python + NumPy 零重型依赖）：
1. 批量 JSON 结构与边界健康体检：
   - 检测时间戳倒挂/重叠、异常短字 (<40ms) / 异常拖音 (>1500ms)、特征缺失、标签非法值及 v0.2 标签覆盖率。
2. 机器初标 vs 人工校对误差剖析：
   - 精确统计字边界 MAE (ms)、RMSE (ms)、Acc@20ms、Acc@50ms；
   - 计算停连 (/ // ^) 与重音 (primary / secondary) 的 Precision / Recall / F1 及误报漏报明细。
3. 双人盲标一致性 (IAA) 与仲裁清单生成：
   - 计算名义 Cohen's Kappa 与序数 Krippendorff's Alpha（停连 B0-B3、重音 S0-S2）；
   - 自动标出低于 0.75 准入阈值的样本，生成交互式「第三人专家仲裁分歧清单 HTML」。
4. 规则阈值与重音权重自适应校准：
   - 基于人工校对后的 JSON 样本，网格搜索最优 short_pause_ms、long_pause_ms、primary_stress_z、
     secondary_stress_z 及重音特征权重 (w_energy, w_duration, w_f0)，直接输出可粘贴回
     D:\\otos\\osot-annotator\\app\\pipeline.py 的推荐配置。
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

STRESS_ORD = {"none": 0, "S0": 0, "secondary": 1, "S1": 1, "primary": 2, "S2": 2}
PAUSE_ORD = {"none": 0, "B0": 0, "connect": 0, "B0_CONNECT": 0, "B1": 1, "short": 2, "B2": 2, "long": 3, "B3": 3}
PAUSE_NOM = {"none": "none", "B0": "none", "connect": "connect", "B0_CONNECT": "connect",
             "B1": "B1", "short": "short", "B2": "short", "long": "long", "B3": "long"}
STRESS_NOM = {"none": "none", "S0": "none", "secondary": "secondary", "S1": "secondary",
              "primary": "primary", "S2": "primary"}


def _extract_tokens(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """兼容 osot-annotator v0.1 JSON 与工具 01 生成的三层解耦 JSON。"""
    if "tokens" in doc and isinstance(doc["tokens"], list):
        return doc["tokens"]
    if "tier1_objective_acoustic" in doc:
        t1 = doc["tier1_objective_acoustic"].get("tokens", [])
        t2 = doc.get("tier2_algorithmic_hypothesis", {}).get("tokens", [])
        t3 = doc.get("tier3_human_gold", {}).get("tokens", [])
        merged = []
        for i, a in enumerate(t1):
            h = t2[i] if i < len(t2) else {}
            g = t3[i] if i < len(t3) else {}
            stress = g.get("human_stress", "UNVERIFIED")
            if stress == "UNVERIFIED":
                stress = h.get("candidate_stress", "S0")
            pause = g.get("human_pause", "UNVERIFIED")
            if pause == "UNVERIFIED":
                pause = h.get("candidate_pause", {}).get("level", "B0")
            merged.append(
                {
                    "index": a.get("index", i),
                    "char": a.get("char", ""),
                    "punct_after": a.get("punct_after", ""),
                    "start_ms": a.get("start_ms", 0),
                    "end_ms": a.get("end_ms", 0),
                    "stress": stress,
                    "pause_after": pause,
                    "features": {
                        "duration_ms": a.get("duration_ms", max(1, a.get("end_ms", 0) - a.get("start_ms", 0))),
                        "gap_after_ms": a.get("gap_after_ms", 0),
                        "energy_db": a.get("energy_db", 0.0),
                        "f0_range_st": a.get("f0_local_span_st", 0.0),
                    },
                    "knowledge_tags": g.get("knowledge_tags", []),
                }
            )
        return merged
    return []


# ---------------------------------------------------------------------------
# 1. JSON 健康体检
# ---------------------------------------------------------------------------
def inspect_single_json(doc: Dict[str, Any], name: str) -> Dict[str, Any]:
    tokens = _extract_tokens(doc)
    issues: List[str] = []
    short_outliers = 0
    long_outliers = 0
    overlap_cnt = 0
    missing_feat = 0
    tagged_v02 = 0

    for i, t in enumerate(tokens):
        s, e = int(t.get("start_ms", 0)), int(t.get("end_ms", 0))
        dur = e - s
        if dur <= 0:
            issues.append(f"字#{i}「{t.get('char','')}」时长非法 ({s}->{e}ms)")
        elif dur < 40:
            short_outliers += 1
        elif dur > 1500:
            long_outliers += 1
        if i > 0 and s < int(tokens[i - 1].get("end_ms", 0)):
            overlap_cnt += 1
        if not isinstance(t.get("features"), dict):
            missing_feat += 1
        if t.get("knowledge_tags"):
            tagged_v02 += 1

    if overlap_cnt:
        issues.append(f"存在 {overlap_cnt} 处相邻字时间戳重叠")
    if short_outliers:
        issues.append(f"存在 {short_outliers} 个极短字 (<40ms)，建议人工复核边界")
    if long_outliers:
        issues.append(f"存在 {long_outliers} 个超长字 (>1500ms)，确认是否为吟诵拖腔或对齐漂移")
    if missing_feat:
        issues.append(f"{missing_feat} 个字缺少 features 声学特征字典")

    stress_dist = Counter(STRESS_NOM.get(str(t.get("stress", "none")), "other") for t in tokens)
    pause_dist = Counter(PAUSE_NOM.get(str(t.get("pause_after", "none")), "other") for t in tokens)

    return {
        "file": name,
        "token_count": len(tokens),
        "issues": issues,
        "is_healthy": len(issues) == 0 and len(tokens) > 0,
        "v02_tag_coverage": round(tagged_v02 / max(1, len(tokens)), 3),
        "stress_distribution": dict(stress_dist),
        "pause_distribution": dict(pause_dist),
    }


# ---------------------------------------------------------------------------
# 2. Cohen's Kappa & 序数 Krippendorff's Alpha
# ---------------------------------------------------------------------------
def cohens_kappa(y1: List[str], y2: List[str]) -> float:
    n = min(len(y1), len(y2))
    if n == 0:
        return 0.0
    y1, y2 = y1[:n], y2[:n]
    labels = sorted(set(y1) | set(y2))
    if len(labels) <= 1:
        return 1.0
    po = sum(1 for a, b in zip(y1, y2) if a == b) / n
    c1, c2 = Counter(y1), Counter(y2)
    pe = sum((c1[l] / n) * (c2[l] / n) for l in labels)
    if abs(1.0 - pe) < 1e-9:
        return 1.0
    return round((po - pe) / (1.0 - pe), 4)


def krippendorff_alpha_ordinal(y1: List[int], y2: List[int]) -> float:
    """两标注员序数 Krippendorff's Alpha（精确按序数秩距公式计算）。"""
    n = min(len(y1), len(y2))
    if n == 0:
        return 0.0
    y1, y2 = y1[:n], y2[:n]
    all_vals = y1 + y2
    cats = sorted(set(all_vals))
    if len(cats) <= 1:
        return 1.0
    freq = Counter(all_vals)

    # 序数距离函数 delta^2(c, k)
    def ord_dist_sq(c: int, k: int) -> float:
        if c == k:
            return 0.0
        lo, hi = (c, k) if c < k else (k, c)
        between = sum(freq[v] for v in cats if lo <= v <= hi)
        g = between - 0.5 * (freq[lo] + freq[hi])
        return g * g

    # 观测分歧 D_o
    do_num = sum(ord_dist_sq(a, b) for a, b in zip(y1, y2))
    D_o = do_num / n

    # 期望分歧 D_e
    N = 2 * n
    de_num = 0.0
    for c in cats:
        for k in cats:
            if c != k:
                de_num += freq[c] * freq[k] * ord_dist_sq(c, k)
    D_e = de_num / (N * (N - 1)) if N > 1 else 0.0
    if D_e < 1e-12:
        return 1.0
    return round(1.0 - (D_o / D_e), 4)


def compute_prf1(y_true: List[str], y_pred: List[str], positive_set: set) -> Dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t in positive_set and p in positive_set)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t not in positive_set and p in positive_set)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t in positive_set and p not in positive_set)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}


# ---------------------------------------------------------------------------
# 3. 对比评估（机标 vs 人工 或 标注员 A vs 标注员 B）
# ---------------------------------------------------------------------------
def compare_pair(doc_a: Dict[str, Any], doc_b: Dict[str, Any], label_a: str = "A", label_b: str = "B") -> Dict[str, Any]:
    tok_a = _extract_tokens(doc_a)
    tok_b = _extract_tokens(doc_b)
    n = min(len(tok_a), len(tok_b))
    if n == 0:
        raise ValueError("待对比的两个 JSON 至少有一个 tokens 为空。")

    start_errs, end_errs = [], []
    s_nom_a, s_nom_b, s_ord_a, s_ord_b = [], [], [], []
    p_nom_a, p_nom_b, p_ord_a, p_ord_b = [], [], [], []
    disagreements: List[Dict[str, Any]] = []

    for i in range(n):
        ta, tb = tok_a[i], tok_b[i]
        se = abs(int(ta["start_ms"]) - int(tb["start_ms"]))
        ee = abs(int(ta["end_ms"]) - int(tb["end_ms"]))
        start_errs.append(se)
        end_errs.append(ee)

        sa_raw, sb_raw = str(ta.get("stress", "none")), str(tb.get("stress", "none"))
        pa_raw, pb_raw = str(ta.get("pause_after", "none")), str(tb.get("pause_after", "none"))

        sa, sb = STRESS_NOM.get(sa_raw, "none"), STRESS_NOM.get(sb_raw, "none")
        pa, pb = PAUSE_NOM.get(pa_raw, "none"), PAUSE_NOM.get(pb_raw, "none")

        s_nom_a.append(sa)
        s_nom_b.append(sb)
        s_ord_a.append(STRESS_ORD.get(sa_raw, 0))
        s_ord_b.append(STRESS_ORD.get(sb_raw, 0))
        p_nom_a.append(pa)
        p_nom_b.append(pb)
        p_ord_a.append(PAUSE_ORD.get(pa_raw, 0))
        p_ord_b.append(PAUSE_ORD.get(pb_raw, 0))

        if sa != sb or pa != pb or max(se, ee) >= 50:
            ctx_lo, ctx_hi = max(0, i - 3), min(n, i + 4)
            ctx = "".join(
                f"【{tok_a[k]['char']}】" if k == i else tok_a[k]["char"] for k in range(ctx_lo, ctx_hi)
            )
            disagreements.append(
                {
                    "index": i,
                    "char": ta.get("char", ""),
                    "context": ctx,
                    "time_ms": f"{ta['start_ms']}–{ta['end_ms']}",
                    "boundary_err_ms": max(se, ee),
                    f"stress_{label_a}": sa,
                    f"stress_{label_b}": sb,
                    f"pause_{label_a}": pa,
                    f"pause_{label_b}": pb,
                }
            )

    all_errs = np.array(start_errs + end_errs, dtype=np.float64)
    mae = float(np.mean(all_errs))
    rmse = float(np.sqrt(np.mean(all_errs ** 2)))
    acc20 = float(np.mean(all_errs <= 20.0))
    acc50 = float(np.mean(all_errs <= 50.0))

    stress_kappa = cohens_kappa(s_nom_a, s_nom_b)
    stress_alpha = krippendorff_alpha_ordinal(s_ord_a, s_ord_b)
    pause_kappa = cohens_kappa(p_nom_a, p_nom_b)
    pause_alpha = krippendorff_alpha_ordinal(p_ord_a, p_ord_b)

    return {
        "compared_tokens": n,
        "boundary_metrics": {
            "mae_ms": round(mae, 2),
            "rmse_ms": round(rmse, 2),
            "acc_at_20ms": round(acc20, 4),
            "acc_at_50ms": round(acc50, 4),
        },
        "iaa_metrics": {
            "stress_cohens_kappa": stress_kappa,
            "stress_krippendorff_alpha_ordinal": stress_alpha,
            "pause_cohens_kappa": pause_kappa,
            "pause_krippendorff_alpha_ordinal": pause_alpha,
            "passes_075_threshold": min(stress_alpha, pause_alpha) >= 0.75,
        },
        "detection_f1_vs_gold": {
            "stress_any": compute_prf1(s_nom_b, s_nom_a, {"primary", "secondary"}),
            "stress_primary": compute_prf1(s_nom_b, s_nom_a, {"primary"}),
            "pause_any": compute_prf1(p_nom_b, p_nom_a, {"short", "long", "B1"}),
            "pause_long": compute_prf1(p_nom_b, p_nom_a, {"long"}),
        },
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
    }


# ---------------------------------------------------------------------------
# 4. 规则阈值与权重自适应校准器（反哺 app/pipeline.py）
# ---------------------------------------------------------------------------
def _zscore(arr: np.ndarray) -> np.ndarray:
    if len(arr) <= 1 or float(np.std(arr)) < 1e-6:
        return np.zeros_like(arr, dtype=np.float64)
    return np.clip((arr - float(np.mean(arr))) / float(np.std(arr)), -2.5, 3.0)


def calibrate_pipeline_thresholds(gold_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    基于人工校对终稿 JSON，自动网格寻优 D:\\otos\\osot-annotator\\app\\pipeline.py 的核心超参：
    - short_pause_ms, long_pause_ms
    - w_eng, w_dur, w_f0r, primary_stress_z, secondary_stress_z
    """
    all_gaps, all_p_gold = [], []
    all_z_eng, all_z_dur, all_z_f0r, all_s_gold = [], [], [], []

    for doc in gold_docs:
        tokens = _extract_tokens(doc)
        if len(tokens) < 3:
            continue
        durs = np.array([t.get("features", {}).get("duration_ms", t["end_ms"] - t["start_ms"]) for t in tokens], dtype=np.float64)
        engs = np.array([t.get("features", {}).get("energy_db", 0.0) for t in tokens], dtype=np.float64)
        f0rs = np.array([t.get("features", {}).get("f0_range_st", 0.0) for t in tokens], dtype=np.float64)
        gaps = np.array([t.get("features", {}).get("gap_after_ms", 0.0) for t in tokens], dtype=np.float64)

        all_z_eng.extend(_zscore(engs).tolist())
        all_z_dur.extend(_zscore(durs).tolist())
        all_z_f0r.extend(_zscore(f0rs).tolist())
        all_gaps.extend(gaps.tolist())
        all_s_gold.extend([STRESS_NOM.get(str(t.get("stress", "none")), "none") for t in tokens])
        all_p_gold.extend([PAUSE_NOM.get(str(t.get("pause_after", "none")), "none") for t in tokens])

    gaps_np = np.array(all_gaps, dtype=np.float64)
    ze = np.array(all_z_eng, dtype=np.float64)
    zd = np.array(all_z_dur, dtype=np.float64)
    zf = np.array(all_z_f0r, dtype=np.float64)

    # 1. 停顿阈值寻优
    best_pause = {"short_pause_ms": 120, "long_pause_ms": 350, "f1_macro": -1.0}
    default_pause_f1 = 0.0
    for sp in range(70, 210, 15):
        for lp in range(max(sp + 100, 240), 520, 25):
            preds = ["long" if g >= lp else ("short" if g >= sp else "none") for g in gaps_np]
            f_s = compute_prf1(all_p_gold, preds, {"short", "long"})["f1"]
            f_l = compute_prf1(all_p_gold, preds, {"long"})["f1"]
            macro = 0.5 * (f_s + f_l)
            if sp == 115 and lp == 340:
                default_pause_f1 = macro
            if macro > best_pause["f1_macro"]:
                best_pause = {"short_pause_ms": sp, "long_pause_ms": lp, "f1_macro": round(macro, 4)}

    # 默认参数基准 (120 / 350)
    def_p_preds = ["long" if g >= 350 else ("short" if g >= 120 else "none") for g in gaps_np]
    default_pause_f1 = round(
        0.5 * (
            compute_prf1(all_p_gold, def_p_preds, {"short", "long"})["f1"]
            + compute_prf1(all_p_gold, def_p_preds, {"long"})["f1"]
        ),
        4,
    )

    # 2. 重音权重与阈值寻优
    weight_grid = [
        (0.45, 0.35, 0.20),  # 原 MVP 默认
        (0.35, 0.40, 0.25),
        (0.30, 0.35, 0.35),
        (0.50, 0.30, 0.20),
        (0.25, 0.50, 0.25),
        (0.40, 0.40, 0.20),
        (0.30, 0.45, 0.25),
    ]
    best_stress = {
        "weights": (0.45, 0.35, 0.20),
        "secondary_stress_z": 0.55,
        "primary_stress_z": 1.15,
        "f1_macro": -1.0,
    }

    def _eval_stress(w: Tuple[float, float, float], s_z: float, p_z: float) -> float:
        sc = w[0] * ze + w[1] * zd + w[2] * zf
        preds = ["primary" if x >= p_z else ("secondary" if x >= s_z else "none") for x in sc]
        f_any = compute_prf1(all_s_gold, preds, {"primary", "secondary"})["f1"]
        f_pri = compute_prf1(all_s_gold, preds, {"primary"})["f1"]
        return 0.5 * (f_any + f_pri)

    default_stress_f1 = round(_eval_stress((0.45, 0.35, 0.20), 0.55, 1.15), 4)
    for w in weight_grid:
        for s_z in np.arange(0.30, 0.95, 0.10):
            for p_z in np.arange(max(s_z + 0.35, 0.85), 1.65, 0.10):
                fm = _eval_stress(w, float(s_z), float(p_z))
                if fm > best_stress["f1_macro"]:
                    best_stress = {
                        "weights": w,
                        "secondary_stress_z": round(float(s_z), 2),
                        "primary_stress_z": round(float(p_z), 2),
                        "f1_macro": round(fm, 4),
                    }

    return {
        "sample_tokens": len(all_s_gold),
        "pause_calibration": {
            "default_params": {"short_pause_ms": 120, "long_pause_ms": 350, "f1_macro": default_pause_f1},
            "recommended_params": best_pause,
        },
        "stress_calibration": {
            "default_params": {
                "weights": [0.45, 0.35, 0.20],
                "secondary_stress_z": 0.55,
                "primary_stress_z": 1.15,
                "f1_macro": default_stress_f1,
            },
            "recommended_params": {
                "weights": list(best_stress["weights"]),
                "secondary_stress_z": best_stress["secondary_stress_z"],
                "primary_stress_z": best_stress["primary_stress_z"],
                "f1_macro": best_stress["f1_macro"],
            },
        },
        "pipeline_py_snippet": (
            f"# 可直接更新至 D:\\otos\\osot-annotator\\app\\pipeline.py -> auto_annotate()\n"
            f"short_pause_ms = {best_pause['short_pause_ms']}\n"
            f"long_pause_ms = {best_pause['long_pause_ms']}\n"
            f"primary_stress_z = {best_stress['primary_stress_z']}\n"
            f"secondary_stress_z = {best_stress['secondary_stress_z']}\n"
            f"scores = {best_stress['weights'][0]} * z_eng + {best_stress['weights'][1]} * z_dur + {best_stress['weights'][2]} * z_f0r\n"
        ),
    }


# ---------------------------------------------------------------------------
# 5. 生成交互式仲裁与校准 HTML 报告
# ---------------------------------------------------------------------------
def export_qa_html(report: Dict[str, Any]) -> str:
    cmp = report.get("comparison", {})
    cal = report.get("calibration", {})
    b = cmp.get("boundary_metrics", {})
    iaa = cmp.get("iaa_metrics", {})
    dis = cmp.get("disagreements", [])

    rows_html = "".join(
        f"<tr><td>#{d['index']+1}</td><td><b>{d['char']}</b></td><td>{d['context']}</td>"
        f"<td>{d['time_ms']}ms</td><td>{d['boundary_err_ms']}ms</td>"
        f"<td>{d.get('stress_A','')} → {d.get('stress_B','')}</td>"
        f"<td>{d.get('pause_A','')} → {d.get('pause_B','')}</td></tr>"
        for d in dis[:100]
    ) or "<tr><td colspan='7' style='text-align:center;color:#94a3b8'>无分歧项</td></tr>"

    snippet = cal.get("pipeline_py_snippet", "").replace("<", "&lt;").replace(">", "&gt;")
    pass_badge = (
        "<span style='color:#10b981;font-weight:700'>✅ 达标 (α ≥ 0.75)</span>"
        if iaa.get("passes_075_threshold")
        else "<span style='color:#f59e0b;font-weight:700'>⚠️ 低于 0.75 准入线，建议第三人仲裁</span>"
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><title>OSOT 标注质检、IAA 一致性与参数校准报告</title>
<style>
  body {{ margin:0; padding:24px; background:#0f172a; color:#f1f5f9; font-family:"PingFang SC","Microsoft YaHei",sans-serif; }}
  .wrap {{ max-width:1120px; margin:0 auto; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin:16px 0; }}
  .kpi {{ background:#1e293b; border:1px solid #334155; border-radius:10px; padding:14px; }}
  .kpi .k {{ font-size:12px; color:#94a3b8; }} .kpi .v {{ font-size:20px; font-weight:700; margin-top:4px; }}
  .panel {{ background:#1e293b; border:1px solid #334155; border-radius:12px; padding:18px; margin-bottom:18px; }}
  pre {{ background:#0b1220; padding:14px; border-radius:8px; overflow-x:auto; color:#7dd3fc; font-size:13px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th,td {{ padding:8px 10px; border-bottom:1px solid #334155; text-align:left; }}
  th {{ color:#94a3b8; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>OSOT 标注质检、双人盲标 IAA 与规则校准报告</h1>
  <div style="color:#94a3b8;font-size:13px">生成时间：{report['generated_at']} ｜ 一致性判定：{pass_badge}</div>

  <div class="grid">
    <div class="kpi"><div class="k">字边界平均误差 MAE</div><div class="v">{b.get('mae_ms','-')} ms</div></div>
    <div class="kpi"><div class="k">边界命中率 Acc@20ms / 50ms</div><div class="v">{round(b.get('acc_at_20ms',0)*100,1)}% / {round(b.get('acc_at_50ms',0)*100,1)}%</div></div>
    <div class="kpi"><div class="k">重音序数 Krippendorff α</div><div class="v">{iaa.get('stress_krippendorff_alpha_ordinal','-')}</div></div>
    <div class="kpi"><div class="k">停连序数 Krippendorff α</div><div class="v">{iaa.get('pause_krippendorff_alpha_ordinal','-')}</div></div>
  </div>

  <div class="panel">
    <h2>自适应参数校准建议（可直接回填至 D:\\otos\\osot-annotator\\app\\pipeline.py）</h2>
    <p style="font-size:13px;color:#cbd5e1">
      停顿 Macro-F1：{cal.get('pause_calibration',{}).get('default_params',{}).get('f1_macro')} →
      <b>{cal.get('pause_calibration',{}).get('recommended_params',{}).get('f1_macro')}</b> ｜
      重音 Macro-F1：{cal.get('stress_calibration',{}).get('default_params',{}).get('f1_macro')} →
      <b>{cal.get('stress_calibration',{}).get('recommended_params',{}).get('f1_macro')}</b>
    </p>
    <pre>{snippet}</pre>
  </div>

  <div class="panel">
    <h2>分歧热点与专家仲裁清单（共 {len(dis)} 处）</h2>
    <table>
      <thead><tr><th>字序</th><th>汉字</th><th>上下文</th><th>时间窗</th><th>边界差</th><th>重音 (A→B)</th><th>停连 (A→B)</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# 6. Demo 自测样例生成器
# ---------------------------------------------------------------------------
def build_demo_pair(out_dir: Path) -> Tuple[Path, Path]:
    chars = list("君不见黄河之水天上来奔流到海不复回")
    auto_toks, gold_toks = [], []
    t = 100
    for i, ch in enumerate(chars):
        dur = 210 if i not in (6, 9, 16) else 380
        gap = 20
        if i == 2:
            gap = 105  # 略低于旧阈值 120ms，人工认为应标短停
        elif i == 9:
            gap = 310  # 略低于旧阈值 350ms，人工认为应标长停
        eng = 68.0 + (7.5 if i in (3, 7, 10, 14) else 0.0)
        f0r = 2.5 + (3.2 if i in (3, 7, 10) else 0.0)

        auto_s = "primary" if i in (3, 7) else ("secondary" if i == 10 else "none")
        gold_s = "primary" if i in (3, 7, 10) else ("secondary" if i == 14 else "none")
        auto_p = "long" if gap >= 350 else ("short" if gap >= 120 else "none")
        gold_p = "long" if i == 9 else ("short" if i == 2 else "none")

        feat = {"duration_ms": dur, "gap_after_ms": gap, "energy_db": eng, "f0_range_st": f0r}
        auto_toks.append({"index": i, "char": ch, "punct_after": "", "start_ms": t, "end_ms": t + dur,
                          "stress": auto_s, "pause_after": auto_p, "features": feat, "knowledge_tags": []})
        shift = 15 if i % 4 == 0 else (-25 if i == 9 else 5)
        gold_toks.append({"index": i, "char": ch, "punct_after": "", "start_ms": t + shift, "end_ms": t + dur + shift,
                          "stress": gold_s, "pause_after": gold_p, "features": feat,
                          "knowledge_tags": ["上山类"] if i == 7 else []})
        t += dur + gap

    p_auto = out_dir / "demo_auto.json"
    p_gold = out_dir / "demo_gold.json"
    p_auto.write_text(json.dumps({"id": "demo_jjj", "title": "将进酒-初标", "tokens": auto_toks}, ensure_ascii=False, indent=2), encoding="utf-8")
    p_gold.write_text(json.dumps({"id": "demo_jjj", "title": "将进酒-人工终稿", "tokens": gold_toks}, ensure_ascii=False, indent=2), encoding="utf-8")
    return p_auto, p_gold


def main() -> None:
    parser = argparse.ArgumentParser(description="OSOT 标注质检、双人盲标 IAA 与规则自适应校准器")
    parser.add_argument("--auto", dest="auto_path", help="机器预标注 JSON（或标注员 A 的 JSON）")
    parser.add_argument("--gold", dest="gold_path", help="人工校对终稿 JSON（或标注员 B 的 JSON）")
    parser.add_argument("--out-dir", dest="out_dir", default="osot_qa_out", help="输出目录")
    parser.add_argument("--demo", action="store_true", help="运行内置 Demo 自测")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        auto_path, gold_path = build_demo_pair(out_dir)
    else:
        if not args.auto_path or not args.gold_path:
            parser.error("请提供 --auto 与 --gold 路径，或使用 --demo 运行自测。")
        auto_path, gold_path = Path(args.auto_path), Path(args.gold_path)

    doc_a = json.loads(auto_path.read_text(encoding="utf-8"))
    doc_b = json.loads(gold_path.read_text(encoding="utf-8"))

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "health_check": [
            inspect_single_json(doc_a, auto_path.name),
            inspect_single_json(doc_b, gold_path.name),
        ],
        "comparison": compare_pair(doc_a, doc_b, label_a="A", label_b="B"),
        "calibration": calibrate_pipeline_thresholds([doc_b]),
    }

    out_json = out_dir / "osot_qa_calibration_report.json"
    out_html = out_dir / "osot_qa_calibration_report.html"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_html.write_text(export_qa_html(report), encoding="utf-8")

    b = report["comparison"]["boundary_metrics"]
    iaa = report["comparison"]["iaa_metrics"]
    cal = report["calibration"]
    print("=== OSOT 质检与校准完成 ===")
    print(f"边界精度: MAE={b['mae_ms']}ms | Acc@20ms={b['acc_at_20ms']*100:.1f}% | Acc@50ms={b['acc_at_50ms']*100:.1f}%")
    print(f"IAA 指标: 重音α={iaa['stress_krippendorff_alpha_ordinal']} (κ={iaa['stress_cohens_kappa']}) | "
          f"停连α={iaa['pause_krippendorff_alpha_ordinal']} (κ={iaa['pause_cohens_kappa']})")
    print(f"分歧热点: {report['comparison']['disagreement_count']} 处")
    print(f"停顿校准: default F1={cal['pause_calibration']['default_params']['f1_macro']} -> "
          f"best F1={cal['pause_calibration']['recommended_params']['f1_macro']} "
          f"(short={cal['pause_calibration']['recommended_params']['short_pause_ms']}ms, "
          f"long={cal['pause_calibration']['recommended_params']['long_pause_ms']}ms)")
    print(f"报告已保存: {out_html}")


if __name__ == "__main__":
    main()
