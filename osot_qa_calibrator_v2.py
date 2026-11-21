#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OSOT 标注质检、双人盲标 IAA 与防过拟合规则校准器 v2.0 (osot_qa_calibrator_v2.py)
================================================================================
v2.0 核心升级（响应 P0/P1 学术评审意见）：
1. 【P0 防过拟合：按说话人划分 Train / Val / Test】
   - 严格按 speaker_id 分组切分（杜绝同一说话人跨集泄漏），仅在 Train+Val 上网格寻优，
     并在从未参与调参的独立 Test 集上报告真实泛化 F1，自动提示过拟合风险！
2. 【P0 严格区分“人工分歧 (iaa)”与“算法错误 (eval)”】
   - --mode iaa  : 双人盲标一致性检验（Cohen's κ + 序数 Krippendorff's α + 0.75 仲裁清单）；
   - --mode eval : 算法预标误差评估（MAE / Acc@20ms / Acc@50ms / Precision / Recall / F1 + FP/FN 拆解）。
3. 【P0 输出独立版本化 calibration.json，绝不直接覆写 app/pipeline.py】
   - 生成独立可回滚、可复现实验的 calibration.json，并附带 5 行安全加载示例。
4. 【P0 统一 osot-unified-v2.0 数据协议】兼容历史 v0.1 单层 JSON 与统一 v2.0 三层 JSON。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_CUR_DIR = Path(__file__).resolve().parent
for _cand in (_CUR_DIR, _CUR_DIR.parent / "00_osot_schema"):
    if (_cand / "osot_schema.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from osot_schema import migrate_to_unified, resolve_effective_tokens  # noqa: E402

STRESS_ORD = {"S0": 0, "S1": 1, "S2": 2}
PAUSE_ORD = {"B0": 0, "B0_CONNECT": 0, "B1": 1, "B2": 2, "B3": 3}


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
    return 1.0 if abs(1.0 - pe) < 1e-9 else round((po - pe) / (1.0 - pe), 4)


def krippendorff_alpha_ordinal(y1: List[int], y2: List[int]) -> float:
    n = min(len(y1), len(y2))
    if n == 0:
        return 0.0
    y1, y2 = y1[:n], y2[:n]
    all_v = y1 + y2
    cats = sorted(set(all_v))
    if len(cats) <= 1:
        return 1.0
    freq = Counter(all_v)

    def d2(c: int, k: int) -> float:
        if c == k:
            return 0.0
        lo, hi = (c, k) if c < k else (k, c)
        g = sum(freq[v] for v in cats if lo <= v <= hi) - 0.5 * (freq[lo] + freq[hi])
        return g * g

    do = sum(d2(a, b) for a, b in zip(y1, y2)) / n
    N = 2 * n
    de = sum(freq[c] * freq[k] * d2(c, k) for c in cats for k in cats if c != k) / (N * (N - 1))
    return 1.0 if de < 1e-12 else round(1.0 - do / de, 4)


def prf1(y_true: List[str], y_pred: List[str], pos: set) -> Dict[str, Any]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t in pos and p in pos)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t not in pos and p in pos)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t in pos and p not in pos)
    pr = tp / (tp + fp) if (tp + fp) else 0.0
    rc = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
    return {"precision": round(pr, 4), "recall": round(rc, 4), "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}


def compare_pair_strict(doc_a: Dict[str, Any], doc_b: Dict[str, Any], mode: str = "eval") -> Dict[str, Any]:
    """
    严格区分两种对比语义：
    - mode='iaa'  : 人工分歧（标注员 A vs 标注员 B）-> 报告 Cohen's κ、序数 Krippendorff's α 与仲裁清单
    - mode='eval' : 算法错误（机器初标 vs 人工终稿）-> 报告边界 MAE/Acc@20ms/50ms 与停连重音 P/R/F1
    """
    ta = resolve_effective_tokens(doc_a, layer="pred" if mode == "eval" else "gold_or_pred")
    tb = resolve_effective_tokens(doc_b, layer="gold_or_pred")
    n = min(len(ta), len(tb))
    if n == 0:
        raise ValueError("待对比 JSON tokens 为空。")

    s_a, s_b, p_a, p_b = [], [], [], []
    errs, diffs = [], []
    for i in range(n):
        a, b = ta[i], tb[i]
        se, ee = abs(int(a["start_ms"]) - int(b["start_ms"])), abs(int(a["end_ms"]) - int(b["end_ms"]))
        errs.extend([se, ee])
        s_a.append(a["stress"])
        s_b.append(b["stress"])
        p_a.append(a["pause_after"])
        p_b.append(b["pause_after"])
        if a["stress"] != b["stress"] or a["pause_after"] != b["pause_after"] or max(se, ee) >= 50:
            diffs.append({
                "index": i, "char": a["char"], "time_ms": f"{a['start_ms']}-{a['end_ms']}",
                "boundary_err_ms": max(se, ee),
                "stress_diff": f"{a['stress']} -> {b['stress']}",
                "pause_diff": f"{a['pause_after']} -> {b['pause_after']}",
                "category": "human_disagreement" if mode == "iaa" else "algorithm_error",
            })

    err_np = np.array(errs, dtype=np.float64)
    out: Dict[str, Any] = {
        "mode": mode,
        "mode_description": "双人盲标人工分歧检验 (IAA)" if mode == "iaa" else "机器初标 vs 人工终稿算法误差评估 (Eval)",
        "compared_tokens": n,
        "boundary_metrics": {
            "mae_ms": round(float(np.mean(err_np)), 2),
            "rmse_ms": round(float(np.sqrt(np.mean(err_np ** 2))), 2),
            "acc_at_20ms": round(float(np.mean(err_np <= 20.0)), 4),
            "acc_at_50ms": round(float(np.mean(err_np <= 50.0)), 4),
        },
        "diff_count": len(diffs),
        "diffs": diffs,
    }
    if mode == "iaa":
        sa = krippendorff_alpha_ordinal([STRESS_ORD.get(x, 0) for x in s_a], [STRESS_ORD.get(x, 0) for x in s_b])
        pa = krippendorff_alpha_ordinal([PAUSE_ORD.get(x, 0) for x in p_a], [PAUSE_ORD.get(x, 0) for x in p_b])
        out["iaa_metrics"] = {
            "stress_cohens_kappa": cohens_kappa(s_a, s_b),
            "stress_krippendorff_alpha_ordinal": sa,
            "pause_cohens_kappa": cohens_kappa(p_a, p_b),
            "pause_krippendorff_alpha_ordinal": pa,
            "arbitration_required_under_075": min(sa, pa) < 0.75,
        }
    else:
        out["algorithm_error_metrics"] = {
            "stress_any": prf1(s_b, s_a, {"S1", "S2"}),
            "stress_primary": prf1(s_b, s_a, {"S2"}),
            "pause_any": prf1(p_b, p_a, {"B1", "B2", "B3"}),
            "pause_long": prf1(p_b, p_a, {"B3"}),
        }
    return out


# ---------------------------------------------------------------------------
# 按说话人分组划分 Train / Val / Test 并生成独立 calibration.json
# ---------------------------------------------------------------------------
def _z(arr: np.ndarray) -> np.ndarray:
    if len(arr) <= 1 or float(np.std(arr)) < 1e-6:
        return np.zeros_like(arr, dtype=np.float64)
    return np.clip((arr - float(np.mean(arr))) / float(np.std(arr)), -2.5, 3.0)


def split_by_speaker(docs: List[Dict[str, Any]], seed: int = 42) -> Dict[str, List[Dict[str, Any]]]:
    """严格按 speaker_id 划分 train / val / test，杜绝同一说话人跨集泄漏。"""
    by_spk: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for d in docs:
        u = migrate_to_unified(d)
        spk = u["meta"].get("speaker_id") or u["meta"].get("id", "spk_unknown")
        by_spk[spk].append(u)

    spks = sorted(by_spk.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(spks)
    ns = len(spks)
    if ns >= 3:
        n_test = max(1, int(round(ns * 0.25)))
        n_val = max(1, int(round(ns * 0.25)))
        n_train = max(1, ns - n_test - n_val)
        train_s = spks[:n_train]
        val_s = spks[n_train : n_train + n_val]
        test_s = spks[n_train + n_val :]
    elif ns == 2:
        train_s, val_s, test_s = [spks[0]], [], [spks[1]]
    else:
        train_s, val_s, test_s = spks, [], []

    def gather(s_list: List[str]) -> List[Dict[str, Any]]:
        res = []
        for s in s_list:
            res.extend(by_spk[s])
        return res

    return {
        "train": gather(train_s),
        "val": gather(val_s),
        "test": gather(test_s),
        "speaker_split": {"train_speakers": train_s, "val_speakers": val_s, "test_speakers": test_s},
    }


def _collect_arrays(docs: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    gaps, ze, zd, zf, s_gold, p_gold = [], [], [], [], [], []
    for d in docs:
        toks = resolve_effective_tokens(d, layer="gold_or_pred")
        if len(toks) < 2:
            continue
        gaps.extend([float(t.get("gap_after_ms", 0.0)) for t in toks])
        ze.extend(_z(np.array([float(t.get("energy_db", 0.0)) for t in toks])).tolist())
        zd.extend(_z(np.array([float(t.get("duration_ms", 200.0)) for t in toks])).tolist())
        zf.extend(_z(np.array([float(t.get("f0_local_span_st", 0.0)) for t in toks])).tolist())
        s_gold.extend([t["stress"] for t in toks])
        p_gold.extend([t["pause_after"] for t in toks])
    return np.array(gaps), np.array(ze), np.array(zd), np.array(zf), s_gold, p_gold


def calibrate_with_heldout_test(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """仅用 train+val 寻优，在独立 test 集上验收，输出独立版本化 calibration.json。"""
    sp = split_by_speaker(docs)
    dev_docs = sp["train"] + sp["val"]
    test_docs = sp["test"]

    g_dev, ze_dev, zd_dev, zf_dev, sg_dev, pg_dev = _collect_arrays(dev_docs)
    if len(sg_dev) == 0:
        raise ValueError("可用于校准的样本数为 0。")

    # 1. 停顿阈值仅在 dev 上寻优
    def eval_pause(g_arr: np.ndarray, pg: List[str], sp_ms: int, lp_ms: int) -> float:
        preds = ["B3" if x >= lp_ms else ("B2" if x >= sp_ms else "B0") for x in g_arr]
        return round(0.5 * (prf1(pg, preds, {"B2", "B3"})["f1"] + prf1(pg, preds, {"B3"})["f1"]), 4)

    best_p = {"short_pause_ms": 120, "long_pause_ms": 350, "dev_f1": -1.0}
    for sp_ms in range(70, 210, 15):
        for lp_ms in range(max(sp_ms + 100, 240), 520, 25):
            f1 = eval_pause(g_dev, pg_dev, sp_ms, lp_ms)
            if f1 > best_p["dev_f1"]:
                best_p = {"short_pause_ms": sp_ms, "long_pause_ms": lp_ms, "dev_f1": f1}

    # 2. 重音权重与阈值仅在 dev 上寻优
    def eval_stress(ze: np.ndarray, zd: np.ndarray, zf: np.ndarray, sg: List[str], w: Tuple[float, float, float], sz: float, pz: float) -> float:
        sc = w[0] * ze + w[1] * zd + w[2] * zf
        preds = ["S2" if x >= pz else ("S1" if x >= sz else "S0") for x in sc]
        return round(0.5 * (prf1(sg, preds, {"S1", "S2"})["f1"] + prf1(sg, preds, {"S2"})["f1"]), 4)

    w_grid = [(0.45, 0.35, 0.20), (0.35, 0.40, 0.25), (0.30, 0.35, 0.35), (0.40, 0.40, 0.20), (0.30, 0.45, 0.25)]
    best_s = {"weights": [0.45, 0.35, 0.20], "secondary_stress_z": 0.55, "primary_stress_z": 1.15, "dev_f1": -1.0}
    for w in w_grid:
        for sz in np.arange(0.30, 0.95, 0.10):
            for pz in np.arange(max(sz + 0.35, 0.85), 1.65, 0.10):
                f1 = eval_stress(ze_dev, zd_dev, zf_dev, sg_dev, w, float(sz), float(pz))
                if f1 > best_s["dev_f1"]:
                    best_s = {
                        "weights": list(w),
                        "secondary_stress_z": round(float(sz), 2),
                        "primary_stress_z": round(float(pz), 2),
                        "dev_f1": f1,
                    }

    # 3. 独立 Test 集验收（绝不参与寻优）
    if test_docs:
        g_te, ze_te, zd_te, zf_te, sg_te, pg_te = _collect_arrays(test_docs)
        test_eval = {
            "heldout_tokens": len(sg_te),
            "pause_default_test_f1": eval_pause(g_te, pg_te, 120, 350),
            "pause_calibrated_test_f1": eval_pause(g_te, pg_te, best_p["short_pause_ms"], best_p["long_pause_ms"]),
            "stress_default_test_f1": eval_stress(ze_te, zd_te, zf_te, sg_te, (0.45, 0.35, 0.20), 0.55, 1.15),
            "stress_calibrated_test_f1": eval_stress(
                ze_te, zd_te, zf_te, sg_te, tuple(best_s["weights"]), best_s["secondary_stress_z"], best_s["primary_stress_z"]
            ),
            "overfit_warning": "",
        }
        if test_eval["stress_calibrated_test_f1"] + 0.12 < best_s["dev_f1"]:
            test_eval["overfit_warning"] = "测试集重音 F1 显著低于开发集（差值 > 0.12），存在过拟合风险，建议扩充不同说话人样本后再启用。"
    else:
        test_eval = {
            "heldout_tokens": 0,
            "overfit_warning": "当前仅有 1 名说话人，无法划分独立测试集！结果仅供探索参考，请至少收集 ≥3 名说话人后再做正式参数冻结。",
        }

    return {
        "calibration_version": f"cal-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "speaker_split": sp["speaker_split"],
        "pause_thresholds": {
            "short_pause_ms": best_p["short_pause_ms"],
            "long_pause_ms": best_p["long_pause_ms"],
        },
        "stress_params": {
            "weights": {"energy": best_s["weights"][0], "duration": best_s["weights"][1], "f0_span": best_s["weights"][2]},
            "secondary_stress_z": best_s["secondary_stress_z"],
            "primary_stress_z": best_s["primary_stress_z"],
        },
        "dev_metrics": {
            "dev_tokens": len(sg_dev),
            "pause_dev_f1": best_p["dev_f1"],
            "stress_dev_f1": best_s["dev_f1"],
        },
        "heldout_test_metrics": test_eval,
        "safe_loader_hint": "在 app/pipeline.py 中读取 calibration.json 而非硬编码覆写，保留版本文件即可随时一键回滚。",
    }


def build_multi_speaker_demo(out_dir: Path) -> List[Path]:
    """构造 4 名说话人（Train/Val/Test 完整分离）的统一格式 Demo 数据。"""
    paths = []
    sentences = [
        ("spk_01", "君不见黄河之水天上来奔流到海不复回"),
        ("spk_02", "世界上没有花的国家是没有的"),
        ("spk_03", "让暴风雨来得更猛烈些吧"),
        ("spk_04", "满以为可以看到辉煌的日出却下起雨来"),
    ]
    for idx_s, (spk, sent) in enumerate(sentences):
        toks = []
        t = 100
        for i, ch in enumerate(sent):
            dur = 340 if i in (3, 7, len(sent) - 1) else 195
            gap = 290 if i == len(sent) // 2 else (100 if i == 2 else 20)
            eng = 74.0 if i in (3, 7) else 66.0
            f0r = 5.0 if i in (3, 7) else 2.0
            toks.append({
                "index": i, "char": ch, "punct_after": "", "start_ms": t, "end_ms": t + dur,
                "stress": "primary" if i in (3, 7) else ("secondary" if i == len(sent) - 1 else "none"),
                "pause_after": "long" if gap >= 260 else ("short" if gap >= 90 else "none"),
                "features": {"duration_ms": dur, "gap_after_ms": gap, "energy_db": eng, "f0_range_st": f0r},
            })
            t += dur + gap
        p = out_dir / f"demo_{spk}.json"
        p.write_text(json.dumps({"id": f"rec_{idx_s+1}", "title": f"演示-{spk}", "speaker_id": spk, "tokens": toks}, ensure_ascii=False, indent=2), encoding="utf-8")
        paths.append(p)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description="OSOT 标注质检、IAA 与防过拟合校准器 v2.0")
    ap.add_argument("--mode", choices=["eval", "iaa", "calibrate"], default="calibrate", help="运行模式：eval(算法误差) / iaa(双人盲标人工分歧) / calibrate(按说话人划分防过拟合校准)")
    ap.add_argument("--auto", help="eval 模式下的机器初标 JSON，或 iaa 模式下的标注员 A JSON")
    ap.add_argument("--gold", help="eval 模式下的专家终稿 JSON，或 iaa 模式下的标注员 B JSON")
    ap.add_argument("--gold-dir", help="calibrate 模式下包含多名说话人终稿 JSON 的目录")
    ap.add_argument("--out-dir", default="osot_qa_v2_out", help="输出目录")
    ap.add_argument("--demo", action="store_true", help="运行内置 4 说话人防过拟合 Demo 自测")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        demo_paths = build_multi_speaker_demo(out_dir)
        docs = [json.loads(p.read_text(encoding="utf-8")) for p in demo_paths]
        cal = calibrate_with_heldout_test(docs)
        cal_path = out_dir / "calibration.json"
        cal_path.write_text(json.dumps(cal, ensure_ascii=False, indent=2), encoding="utf-8")
        print("=== OSOT 工具 02 v2.0 防过拟合校准 Demo 完成 ===")
        print(f"说话人划分 : {cal['speaker_split']}")
        print(f"开发集 F1  : pause={cal['dev_metrics']['pause_dev_f1']} | stress={cal['dev_metrics']['stress_dev_f1']}")
        te = cal["heldout_test_metrics"]
        print(f"独立测试集 : pause {te['pause_default_test_f1']} -> {te['pause_calibrated_test_f1']} | stress {te['stress_default_test_f1']} -> {te['stress_calibrated_test_f1']}")
        print(f"独立配置已保存: {cal_path}")
        return

    if args.mode in ("eval", "iaa"):
        if not args.auto or not args.gold:
            ap.error(f"--mode {args.mode} 需要提供 --auto 与 --gold。")
        da = json.loads(Path(args.auto).read_text(encoding="utf-8"))
        db = json.loads(Path(args.gold).read_text(encoding="utf-8"))
        res = compare_pair_strict(da, db, mode=args.mode)
        out_p = out_dir / f"osot_{args.mode}_report.json"
        out_p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"=== {res['mode_description']} 完成 -> {out_p} ===")
        print(json.dumps({k: v for k, v in res.items() if k != "diffs"}, ensure_ascii=False, indent=2))
    else:
        if not args.gold_dir:
            ap.error("--mode calibrate 需要提供 --gold-dir（或使用 --demo）。")
        docs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(args.gold_dir).glob("*.json"))]
        cal = calibrate_with_heldout_test(docs)
        cal_path = out_dir / "calibration.json"
        cal_path.write_text(json.dumps(cal, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"=== 独立参数文件已生成 -> {cal_path} ===")
        print(json.dumps(cal["heldout_test_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
