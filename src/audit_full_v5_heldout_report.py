"""Recalculate all saved predictions independently and export evidence report."""
from __future__ import annotations

from pathlib import Path
import json
import hashlib
import statistics

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "checkpoints/fgclip2_full_v5_heldout_20260914"


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    result = read(OUT / "test_summary.json")
    plan = read(OUT / "evaluation_plan.json")
    efficiency = read(OUT / "efficiency/efficiency_summary.json")
    index = read(OUT / "prediction_index.json")
    recalc = []
    for row in result["rows"]:
        scores = []
        entry = {k: row[k] for k in ("variant", "seed")}
        for lang in ("zh", "en"):
            path = OUT / "predictions" / f"{row['variant']}_seed_{row['seed']}_{lang}_test.npz"
            assert sha(path) == index[path.name]
            a = np.load(path, allow_pickle=False)
            frame = pd.read_csv(ROOT / f"data_submission_v1/{lang}/test.csv")
            np.testing.assert_array_equal(a["row_indices"], np.arange(len(frame)))
            np.testing.assert_array_equal(a["labels"], frame.label.values)
            pred = a["logits"].argmax(1)
            f1 = f1_score(a["labels"], pred, labels=[0, 1, 2], average="macro", zero_division=0)
            acc = accuracy_score(a["labels"], pred)
            cm = confusion_matrix(a["labels"], pred, labels=[0, 1, 2])
            assert abs(f1 - row[lang]["macro_f1"]) < 1e-12
            assert abs(acc - row[lang]["accuracy"]) < 1e-12
            np.testing.assert_array_equal(cm, row[lang]["confusion_counts"])
            np.testing.assert_allclose(cm / cm.sum(1, keepdims=True), row[lang]["confusion_row_normalized"])
            scores.append(f1)
            entry.update({f"{lang}_mf1": f1, f"{lang}_accuracy": acc, f"{lang}_n": len(frame)})
        entry["lb_mf1"] = sum(scores) / 2
        assert abs(entry["lb_mf1"] - row["lb_mf1"]) < 1e-12
        recalc.append(entry)
    for variant in ("full_v5", "no_cross_control"):
        values = [r["lb_mf1"] for r in recalc if r["variant"] == variant]
        assert abs(statistics.mean(values) - result[variant]["lb_mf1"]["mean"]) < 1e-12
        assert abs(statistics.stdev(values) - result[variant]["lb_mf1"]["sd"]) < 1e-12
    raw = read(OUT / "efficiency/timing_raw.json")
    assert len(raw) == 40 and all(len(r["request_ms"]) == 64 for r in raw)
    assert sha(ROOT / "profile_full_v5_locked.py") == efficiency["protocol"]["profiler_sha256"]
    for row in raw:
        mean = statistics.mean(row["request_ms"])
        assert abs(mean - row["latency_ms_per_batch"]) < 1e-10
        assert abs(1000 * row["batch_size"] / mean - row["throughput_samples_per_second"]) < 1e-8
        assert abs(mean / row["batch_size"] - row["latency_ms_per_sample_amortized"]) < 1e-10
    for summary in efficiency["rows"]:
        matched = [r for r in raw if all(r[k] == summary[k] for k in ("scope", "variant", "batch_size"))]
        assert len(matched) == 5
        for key in ("latency_ms_per_batch", "latency_ms_per_sample_amortized", "throughput_samples_per_second"):
            values = [r[key] for r in matched]
            assert abs(statistics.mean(values) - summary[key]["mean"]) < 1e-9
            assert abs(statistics.stdev(values) - summary[key]["sd"]) < 1e-9
        assert summary["peak_allocated_mib"] == max(r["peak_allocated_mib"] for r in matched)
    # Verify tuple-level equality, not only marginal column equality.
    manifest = pd.read_csv(ROOT / "data_submission_v1/split_manifest.csv").fillna("")
    originals = []
    coverage = {}
    for lang, directory in (("zh", "data_cisd"), ("en", "data")):
        original = pd.concat([pd.read_csv(ROOT / directory / f"{s}.csv") for s in ("train", "val")]).fillna("")
        normalize_path = lambda p: str(p).replace("\\", "/").lower()
        keys = lambda f: set(zip(f.image_path.map(normalize_path), f.text.astype(str), f.label.astype(int)))
        available = keys(original)
        known = keys(manifest[manifest.language == lang])
        coverage[lang] = dict(original_csv_rows=len(original), original_unique_pairs=len(available),
                              pairs_outside_existing_manifest=len(available - known))
        for partition in ("train", "val", "test"):
            f = pd.read_csv(ROOT / f"data_submission_v1/{lang}/{partition}.csv").fillna("")
            expected = manifest[(manifest.language == lang) & (manifest.split == partition)]
            assert keys(f) == keys(expected) and len(f) == len(expected)
    for row in plan["models"]:
        assert sha(row["checkpoint"]) == row["checkpoint_sha256"]
    for name, expected in plan["protected_cache_stamps"].items():
        p = Path(name)
        assert [p.stat().st_size, p.stat().st_mtime_ns] == expected
    pd.DataFrame(recalc).to_csv(OUT / "test_results.csv", index=False)
    flat = []
    for row in efficiency["rows"]:
        f = {k: row[k] for k in ("scope", "variant", "batch_size", "peak_allocated_mib")}
        for name in ("latency_ms_per_batch", "latency_ms_per_sample_amortized", "throughput_samples_per_second"):
            f[name + "_mean"] = row[name]["mean"]
            f[name + "_sd"] = row[name]["sd"]
        flat.append(f)
    pd.DataFrame(flat).to_csv(OUT / "efficiency/efficiency_results.csv", index=False)
    audit = dict(passed=True, prediction_files_recalculated=len(index), timing_requests_recalculated=40*64,
                 sklearn_metrics_agree=True, means_and_sample_sd_verified=True, checkpoint_and_original_cache_unchanged=True,
                 current_dataset_inventory=coverage, independent_confirmatory_test_available=False,
                 test_summary_sha256=sha(OUT / "test_summary.json"), efficiency_summary_sha256=sha(OUT / "efficiency/efficiency_summary.json"),
                 auditor_sha256=sha(Path(__file__)))
    (OUT / "independent_recalculation_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    generate_report(result, efficiency, audit)
    print(json.dumps(audit, indent=2))


def generate_report(result, efficiency, audit):
    fmt = lambda x, digits=4: f"{x['mean']:.{digits}f} ± {x['sd']:.{digits}f}"
    selected = result["deployment"]
    paired_control = next(r for r in result["rows"] if r["seed"] == 2027 and r["variant"] == "no_cross_control")
    full_timing = {r["batch_size"]: r for r in efficiency["rows"] if r["variant"] == "full_v5" and r["scope"] == "backbone_plus_head"}
    lines = ["# Full v5 留出测试与效率实测报告", "", "测量日期：2026-09-14", "",
             "## 结论", "",
             "已完成锁定 Full-v5 与配对 No-cross 的五种子留出集测试，以及同条件推理速度和显存实测。没有重新训练、修改原检查点或降低 No-cross 的配置。",
             "", "**独立性限制：现有 test 集结果早已在原始消融中被查看，不能称为全新独立盲测。当前结果是 retrospective held-out evaluation（回顾性留出评估）。真正未查看的新测试数据尚缺。**", "",
             f"五种子平均 LB-MF1：Full {fmt(result['full_v5']['lb_mf1'])}；No-cross {fmt(result['no_cross_control']['lb_mf1'])}。Full 平均值较低，不能宣称总体优于 No-cross。",
             f"预先按验证集选定的部署模型保持 seed 2027、epoch 4、α=0.5：Full LB-MF1={selected['lb_mf1']:.4f}，配对 No-cross={paired_control['lb_mf1']:.4f}。没有按测试集重新选择种子、epoch 或 α。", "",
             "## 数据与锁定协议", "",
             "- CISD：3,883 个 test 样本；MVSA-Single：263 个 test 样本。标签顺序为 Negative、Neutral、Positive。",
             "- 五个训练种子：42、123、3407、2026、2027；所有 Full 均固定 α=0.5。",
             "- 指标：先分别计算两种语言的三类别 Macro-F1，再等权平均得到 LB-MF1。SD 使用 ddof=1，是训练随机性波动，不是总体置信区间。",
             "- 无测试集调参、阈值校准、类别权重修改或样本剔除。没有进行确认性显著性检验。",
             "- 分组清单和六份 CSV 匹配，已记录的重复组跨 train/val/test 数量为 0；这不等于保证不存在任何未检出的语义近重复。",
             "- 原 test 曾用于 15 个历史消融；内部 probe 也已评估，而且属于当前所选 Full-v5 训练分区，不能重新命名为独立测试。",
             "- 保留全部现有缓存；仅新增 4,146 条 test 文本的 contextual token 特征。", "",
             "## 五种子测试结果", "",
             "| Model | CISD Acc | CISD Macro-F1 | MVSA Acc | MVSA Macro-F1 | LB-MF1 |",
             "|---|---:|---:|---:|---:|---:|"]
    for variant, label in (("no_cross_control", "No-cross control"), ("full_v5", "Full-v5 α=0.5")):
        r = result[variant]
        lines.append(f"| {label} | {fmt(r['zh_accuracy'])} | {fmt(r['zh_macro_f1'])} | {fmt(r['en_accuracy'])} | {fmt(r['en_macro_f1'])} | {fmt(r['lb_mf1'])} |")
    lines += ["", "| Seed | Full LB-MF1 | No-cross LB-MF1 | Full − No-cross |", "|---:|---:|---:|---:|"]
    for seed in (42, 123, 3407, 2026, 2027):
        f = next(r["lb_mf1"] for r in result["rows"] if r["seed"] == seed and r["variant"] == "full_v5")
        n = next(r["lb_mf1"] for r in result["rows"] if r["seed"] == seed and r["variant"] == "no_cross_control")
        lines.append(f"| {seed} | {f:.6f} | {n:.6f} | {f-n:+.6f} |")
    lines += ["", "Full 为 1 胜、2 平、2 负。单个部署种子的优势不能替代五种子总体结论。", "",
              "## 效率实测", "",
              "RTX 5080；Windows WDDM；PyTorch 2.9.1+cu128；FP32 参数，FP16 autocast，Full 最后一步残差相加为 FP32。主干和分类头均为 eval 模式。",
              "每种条件 5 轮、每轮 20 次预热和 64 次正式请求。每次请求前后同步 CUDA；计时包含模型 CPU 调度，不包含图片读取、预处理、分词、内容 mask 构造及 CPU→GPU 传输。Full/No-cross 交替测试顺序，使用相同的按行取样验证输入且中英文等量。",
              "推理计时的 ± 为五轮技术重复的样本 SD，不是训练种子 SD。显存是进程内 PyTorch 峰值 allocated memory，不是 nvidia-smi 显示的整卡占用。桌面程序共用该 GPU，未锁频。", "",
              "### 主干加分类头", "", "| Model | Batch | 每批延迟 ms | 摊销每样本 ms | samples/s | 峰值显存 MiB |", "|---|---:|---:|---:|---:|---:|"]
    for r in efficiency["rows"]:
        if r["scope"] == "backbone_plus_head":
            lines.append(f"| {r['variant']} | {r['batch_size']} | {fmt(r['latency_ms_per_batch'],2)} | {fmt(r['latency_ms_per_sample_amortized'],3)} | {fmt(r['throughput_samples_per_second'],1)} | {r['peak_allocated_mib']:.1f} |")
    lines += ["", "Batch 16 的每样本时间是每批时间除以 16，不能当作单请求响应延迟。上述范围也不是包含解码、传输和服务开销的完整应用端到端延迟。", "",
              "### 仅缓存分类头", "", "| Model | Batch | 每批延迟 ms | samples/s | 峰值显存 MiB |", "|---|---:|---:|---:|---:|"]
    for r in efficiency["rows"]:
        if r["scope"] == "cached_head_only":
            lines.append(f"| {r['variant']} | {r['batch_size']} | {fmt(r['latency_ms_per_batch'],2)} | {fmt(r['throughput_samples_per_second'],1)} | {r['peak_allocated_mib']:.1f} |")
    p = efficiency["parameters"]
    lines += ["", "### 参数量", "", "| Model | 主干参数 | 可训练头参数 | 总参数 |", "|---|---:|---:|---:|"]
    for v in ("no_cross_control", "full_v5"):
        h = p["head_parameters"][v]
        lines.append(f"| {v} | {p['backbone_parameters']:,} | {h:,} | {p['backbone_parameters']+h:,} |")
    lines += ["", "可训练头参数表示训练阶段的结构参数量，不是 eval 模式下 requires_grad=False 的临时状态。Full 增加 663,300 个分支参数。未估算或编造 FLOPs。", "",
              "### 已保存训练耗时", "",
              "这些数值来自各次真实训练日志，包含 refinement 阶段实际执行的训练与验证（包括 epoch 0；各运行实际轮数可能不同），不含父 No-cross 模型训练、特征提取及全部超参探索，也不是新做的同条件训练速度比较。", "",
              "| Seed | Full refinement 秒 | No-cross refinement 秒 |", "|---:|---:|---:|"]
    times = efficiency["historical_refinement_times"]["rows"]
    for seed in (42, 123, 3407, 2026, 2027):
        f = next(r["seconds"] for r in times if r["seed"] == seed and r["variant"] == "full_v5")
        n = next(r["seconds"] for r in times if r["seed"] == seed and r["variant"] == "no_cross_control")
        lines.append(f"| {seed} | {f:.2f} | {n:.2f} |")
    lines += ["", "## 可放入论文的英文结果段落", "",
              f"The locked Full-v5 model was evaluated retrospectively on the fixed held-out partitions of CISD (n = 3,883) and MVSA-Single (n = 263). Across five training seeds, Full-v5 achieved an LB-MF1 of {fmt(result['full_v5']['lb_mf1'])}, compared with {fmt(result['no_cross_control']['lb_mf1'])} for the paired No-cross control (mean ± sample s.d.). Full-v5 exceeded the control in one seed, tied in two, and underperformed in two. Thus, the validation advantage did not translate into a higher mean held-out score. The deployment checkpoint, selected previously using validation performance (seed 2027, epoch 4, α = 0.5), achieved an LB-MF1 of {selected['lb_mf1']:.4f}. All checkpoints and inference scales were fixed before this evaluation. Because results on these test partitions had been inspected in earlier development, this evaluation should not be interpreted as a newly untouched confirmatory test.", "",
              f"On an NVIDIA GeForce RTX 5080, the locked Full-v5 backbone-plus-head implementation required {fmt(full_timing[1]['latency_ms_per_batch'],2)} ms per single-sample request. At batch size 16, the mean amortized inference time was {fmt(full_timing[16]['latency_ms_per_sample_amortized'],3)} ms per sample, corresponding to {fmt(full_timing[16]['throughput_samples_per_second'],1)} samples/s. Timing used five technical rounds, each with 20 warm-up requests and 64 measured requests, with CUDA synchronization. Image decoding, preprocessing, tokenization and host-to-device transfer were excluded. Full-v5 contains 3,765,513 head parameters and 387,568,907 total parameters, including the frozen backbone.", "",
              "## 复现与校验", "",
              "在项目目录使用 peixun 环境执行：", "", "```powershell",
              "python evaluate_full_v5_heldout.py --preflight", "python evaluate_full_v5_heldout.py",
              "python profile_full_v5_locked.py", "python audit_full_v5_heldout_report.py",
              "python -m unittest test_full_v5_heldout_reporting -v", "```", "",
              "评估会复用已完成 test 文本和预测；测速命令会重新测量并覆盖该轮测速报告，若要保留旧测速请给 --output 指定新的目录。",
              "", "- evaluation_plan.json：评估前锁定的模型路径、哈希、种子、α 和指标规则。",
              "- predictions/*.npz：20 份逐样本 logits、真实标签和行号。",
              "- test_results.csv、test_summary.json：逐种子指标与混淆矩阵。",
              "- efficiency/timing_raw.json：2,560 次请求的原始计时。",
              "- efficiency/efficiency_results.csv：同条件测速汇总。",
              "- independent_recalculation_audit.json：使用 sklearn 从逐样本预测重新计算指标的校验。这里的 independent 指计算复核，不代表测试集独立。",
              "", "## 尚未完成的独立盲测", "",
              "需要一批未参与训练、验证、模型探索或结果查看的带标签图文样本；对新样本执行与现有全部数据的去重后，先锁定协议，再一次性评估，不能看结果后继续调参。当前原始 CSV 中没有清单之外的新样本可直接使用。不要把已训练样本重新划分，或把已看过的 test/probe 改名后当作新独立集。", ""]
    report_path = ROOT / "paper_revision/FullV5_Test_and_Efficiency_Report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"REPORT: {report_path}")


if __name__ == "__main__":
    main()
