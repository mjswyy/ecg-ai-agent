#!/usr/bin/env python3
"""
Preprocess Data — ECGFounder 官方协议版（5000 样本 / 12 导联 / 10s@500Hz）

用途（里程碑 M0.1）:
    从原始 WFDB 数据重建 ECGFounder 兼容的预处理数据，输出到 processed_5k/。

与旧版 preprocess_data.py（4096 版）的差异:
    1. 目标长度 5000 样本（10 秒 @ 500Hz），不截断 10 秒记录
    2. 滤波采用 ECGFounder 官方配方（util.filter_bandpass 的逐通道实现）:
        50Hz 陷波(Q=30) → Butterworth 4阶带通 [0.67, 40]Hz → 0.4s 中值滤波去基线漂移
       （官方在 500Hz 下硬编码 fs；此处按各数据源原生采样率设计滤波器，再重采样到 500Hz）
    3. 不做 5σ 异常值裁剪（官方无此步骤）
    4. z-score 采用官方风格: 全信号（12 导联合并）统计 mean/std（仅统计非零填充区域）
    5. 导联顺序校验: 按 .hea 实际 sig_name 重排为 I,II,III,aVR,aVL,aVF,V1-V6
    6. 数据划分复用 --old-manifest-dir（默认 processed/）的 train/val/test manifest

划分现状声明（第三轮审查 🔴-2/🔴-3）:
    - processed/ 目录 = 历史记录级划分产物（30167/6462/6472，70/15/15 按记录切分，
      同一患者多次心电可能跨划分），仅作本脚本的历史划分来源，论文与评测一律不用；
    - processed_5k/ 的权威划分 = 患者级 80/10/10（30232/3763/3754），由
      scripts/patient_level_split.py 维护（幂等，含权威锚点断言）。
    - 重跑本脚本后必须重跑 patient_level_split.py 恢复患者级划分，
      否则 manifest 会退回记录级并造成跨数据集混用泄漏。

用法:
    # 冒烟测试
    python scripts/preprocess_data_5k.py --max-records 10
    # 全量
    python scripts/preprocess_data_5k.py
    # 断点续跑
    python scripts/preprocess_data_5k.py --skip-existing
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.loader import ECGLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# ECGFounder 官方滤波配方（adaptation: fs 按原生采样率传入）
# ============================================================

STANDARD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
                  "V1", "V2", "V3", "V4", "V5", "V6"]


def filter_bandpass(signal: np.ndarray, fs: float) -> np.ndarray:
    """ECGFounder 官方预处理滤波（对应 util.filter_bandpass）。

    官方在 500Hz 硬编码; 此处按原生 fs 设计滤波器, 重采样在调用方完成。

    步骤（与官方一致）:
        1) 50Hz 陷波 (iirnotch, Q=30), 逐通道零相位滤波
        2) Butterworth 4阶带通 [0.67, 40] Hz, 逐通道零相位滤波
        3) 0.4s 窗口中值滤波提取基线漂移, 减去

    Args:
        signal: (channels, time) 浮点数组（可含 NaN，本函数先做 nan_to_num）
        fs: 原生采样率 (Hz)

    Returns:
        滤波后信号, 与输入同形状
    """
    from scipy.signal import iirnotch, filtfilt, butter, medfilt

    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    # 1) 工频陷波 50Hz, Q=30（3F-💡-1：官方配方固定 50Hz；
    #    Georgia 为 60Hz 市电源，此处未单独处理，与官方预处理严格对齐）
    b, a = iirnotch(50.0, 30.0, fs)
    out = np.zeros_like(signal, dtype=np.float64)
    for c in range(signal.shape[0]):
        out[c] = filtfilt(b, a, signal[c])

    # 2) 带通 [0.67, 40] Hz, 4阶
    b, a = butter(N=4, Wn=[0.67, 40.0], btype="bandpass", fs=fs)
    for c in range(signal.shape[0]):
        out[c] = filtfilt(b, a, out[c])

    # 3) 中值滤波基线漂移去除 (kernel = 0.4s, 保证奇数)
    kernel = int(0.4 * fs) + 1
    if kernel % 2 == 0:
        kernel += 1
    baseline = np.zeros_like(out)
    for c in range(signal.shape[0]):
        baseline[c] = medfilt(out[c], kernel_size=kernel)
    out = out - baseline

    return out.astype(np.float32)


def resample_to_500(ecg: np.ndarray, orig_fs: float) -> np.ndarray:
    """整段精确重采样到 500Hz（检查报告 1.6 修复）。

    旧实现用"防爆内存"循环反复折半 up/down：257Hz（500/257 无法用小因子
    表示）比率被钳成 63/33（1.9455→1.9091），信号被时间压缩、尾部静默补零
    （5000 样本中 93 个为 0）并污染 z-score。
    修复：直接使用真实比率 resample_poly。scipy≥1.14 的 upfirdn 已统一为
    polyphase 实现（内存 O(L)、每次输出 O(N/P)），大因子既不会爆内存也不
    需要钳制——钳制循环是历史包袱，删除即可。
    """
    from scipy import signal as sp_signal
    from math import gcd

    if abs(orig_fs - 500.0) < 1e-3:
        return ecg

    target_len = round(ecg.shape[1] * 500.0 / orig_fs)
    up, down = 500, int(orig_fs)
    g = gcd(up, down)
    up //= g
    down //= g

    out = np.zeros((ecg.shape[0], target_len), dtype=np.float32)
    for i in range(ecg.shape[0]):
        y = sp_signal.resample_poly(ecg[i].astype(np.float64), up, down)
        n = min(len(y), target_len)
        out[i, :n] = y[:n]
    return out


def segment_to_5000(ecg: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """居中裁剪 / 对称零填充到 5000 样本。返回 (seg, pad_left, pad_right)。"""
    target = 5000
    L = ecg.shape[1]
    if L == target:
        return ecg, 0, 0
    if L > target:
        start = (L - target) // 2
        return ecg[:, start:start + target], 0, 0
    pad_total = target - L
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    padded = np.pad(ecg, ((0, 0), (pad_left, pad_right)),
                    mode="constant", constant_values=0.0)
    return padded, pad_left, pad_right


def official_zscore(ecg: np.ndarray, pad_left: int, pad_right: int) -> np.ndarray:
    """官方风格 z-score: 全信号合并统计（单标量 mean/std），仅统计非填充区域。

    第三轮审查 3B-DP-T2 修复：归一化后把填充区显式置零——
    旧版对全信号（含填充区）做 (x-mean)/std，零填充区变成非零常量
    （约 -mean/std），破坏"零填充"语义（实测 48 条短记录受影响）。
    """
    L = ecg.shape[1]
    valid = ecg[:, pad_left:(L - pad_right if pad_right > 0 else L)]
    mean = float(np.mean(valid))
    std = float(np.std(valid)) + 1e-8
    out = ((ecg - mean) / std).astype(np.float32)
    if pad_left > 0:
        out[:, :pad_left] = 0.0
    if pad_right > 0:
        out[:, L - pad_right:] = 0.0
    return out


def reorder_leads(signal: np.ndarray, lead_names: List[str]) -> Tuple[np.ndarray, bool]:
    """按 .hea 实际导联名重排为 I,II,III,aVR,aVL,aVF,V1-V6。

    Returns:
        (重排后信号, 是否发生了重排)
    """
    if signal.shape[0] != 12:
        return signal, False
    # 大小写归一
    names = [str(n).strip().upper() for n in lead_names]
    target = [n.upper() for n in STANDARD_LEADS]
    if names == target:
        return signal, False
    try:
        idx = [names.index(t) for t in target]
    except ValueError:
        # 4C 审查修复：💡-9 —— 匹配失败时显式告警并说明风险：保持原导联顺序
        # 会把 aVR/aVL/V1-V6 等按错误顺序送入模型。
        logger.warning(f"导联名无法匹配标准顺序: {names}，保持原顺序"
                       f"（可能以错误导联序送入模型）")
        return signal, False
    return signal[idx, :], True


# ============================================================
# 主流程
# ============================================================

def _default_raw_dir():
    """3F-Y5 修复：探测原始数据目录（旧版硬编码互不一致的绝对路径
    Desktop\\ECG vs Desktop\\ecg资料，换机即失效）。探测失败返回 None，
    由调用方要求显式传 --raw-dir。"""
    repo = Path(__file__).parent.parent
    stem = "classification-of-12-lead-ecgs-the-physionetcomputing-" \
           "in-cardiology-challenge-2020-1.0.2"
    candidates = [
        repo.parent / stem / "training",
        Path.home() / "Desktop" / "ecg资料" / stem / "training",
        Path.home() / "Desktop" / "ECG" / stem / "training",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def main():
    parser = argparse.ArgumentParser(description="ECGFounder 协议预处理 (5000 样本)")
    parser.add_argument(
        "--raw-dir", type=str,
        default=_default_raw_dir(),
        help="原始训练数据目录（不传时自动探测；探测失败必须显式指定）",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="data/physionet2020/processed_5k",
        help="输出目录（新 5000 样本版）",
    )
    parser.add_argument(
        "--old-manifest-dir", type=str,
        default="data/physionet2020/processed",
        help="旧版 processed 目录（复用其 train/val/test 划分）",
    )
    parser.add_argument(
        "--max-records", type=int, default=None,
        help="最大处理条数（冒烟测试用）",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="跳过已存在的输出文件（断点续跑）",
    )
    args = parser.parse_args()
    # 4C 审查修复：💡-10 —— 冒烟模式（--max-records）只测计算链路，不落盘 npy，
    # 避免在 processed_5k/ 遗留无 manifest 引用的孤儿文件（历史 repair_manifest_5k.py
    # 即为同类跳过逻辑的补丁）。
    smoke = args.max_records is not None

    if not args.raw_dir:
        sys.exit("未探测到原始数据目录——请用 --raw-dir 显式指定（3F-Y5）")
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    old_dir = Path(args.old_manifest_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 读取旧划分，构建 (source, record_id) -> split 映射 ----
    # 第三轮审查 🔴-2/🔴-3：processed/ 为历史记录级产物（70/15/15 按记录切分）；
    # 患者级权威划分由 patient_level_split.py 维护——重跑本脚本后必须重跑它恢复。
    split_map: Dict[Tuple[str, str], str] = {}
    for split in ("train", "val", "test"):
        mp = old_dir / f"{split}_manifest.json"
        if not mp.exists():
            logger.error(f"找不到旧划分文件: {mp}")
            sys.exit(1)
        with open(mp, encoding="utf-8") as f:
            data = json.load(f)
        for item in data["files"]:
            split_map[(item["source"], item["record_id"])] = split
    logger.info(f"旧划分加载: {len(split_map)} 条记录")

    # ---- 2. 初始化加载器 ----
    loader = ECGLoader(raw_dir)

    # ---- 3. 处理 ----
    manifest: List[Dict] = []
    skipped = 0
    reordered = 0
    unmatched = 0
    unknown_split = 0

    # 与 loader.iter_records 相同的确定性顺序（sorted sources + sorted .hea）
    total = loader.get_record_count()
    logger.info(f"共发现 {total} 条记录，开始处理...")

    pbar = tqdm(total=min(total, args.max_records or total), desc="Processing 5k")
    count = 0

    for source in loader.sources:
        source_dir = raw_dir / source
        for hea_file in sorted(source_dir.rglob("*.hea")):
            if hea_file.name.startswith("."):
                continue
            if args.max_records and count >= args.max_records:
                break

            sample = loader.load_record(str(hea_file))
            if sample is None:
                skipped += 1
                continue
            count += 1
            pbar.update(1)

            key = (sample.source, sample.record_id)
            if key not in split_map:
                unknown_split += 1
                logger.warning(f"{key} 不在旧划分中，跳过")
                continue

            signal_file = f"{sample.source}_{sample.record_id}.npy"
            signal_path = output_dir / signal_file
            # 断点续跑：文件已存在时跳过计算，但仍需在 manifest 中记录（否则划分缺条目）
            file_exists = args.skip_existing and signal_path.exists()

            try:
                # 复检1 修复（🔴）：leads_ok/leads_reordered 必须在 if/else 前初始化——
                # 旧版只在 else 分支定义，断点续跑（file_exists=True）分支在
                # manifest.append 引用时 NameError（异常被 except 捕获计入 skipped，
                # 所有已存在记录缺失于 manifest）
                leads_ok = True
                leads_reordered = False
                if file_exists:
                    # 复用已有文件，只补充 manifest 条目（导联状态以首次处理为准，
                    # 复用文件视为已处理：leads_matched=True / leads_reordered=False）
                    normalized = np.load(signal_path)
                    if normalized.shape[0] != 12 and normalized.shape[1] == 12:
                        normalized = normalized.T
                else:
                    # 导联重排
                    signal, did_reorder = reorder_leads(sample.signal, sample.lead_names)
                    if did_reorder:
                        reordered += 1
                    # 4C 复查（β）：导联不匹配时在 manifest 留痕。复检1 修正语义：
                    # leads_reordered = 是否发生了重排；leads_matched = 导联序是否
                    # 为标准序（False 即"匹配失败保持原序"，下游据此排查）。
                    # 5C 审查（💡-1）：reorder_leads 大小写不敏感比较，此处必须
                    # 同样归一化。复检A（🟠-2）：右侧 STANDARD_LEADS 含 aVR 等
                    # 混合大小写，两侧都要 upper（旧版只归一左侧恒为 False）
                    leads_ok = ([str(n).strip().upper() for n in (sample.lead_names or [])]
                                == [n.upper() for n in STANDARD_LEADS])
                    leads_reordered = did_reorder

                    # 官方滤波（原生采样率）→ 重采样 500Hz → 分段 5000
                    filtered = filter_bandpass(signal, sample.fs)
                    resampled = resample_to_500(filtered, sample.fs)
                    seg, pad_l, pad_r = segment_to_5000(resampled)
                    normalized = official_zscore(seg, pad_l, pad_r)
                    # 4C 审查修复：💡-10 —— 冒烟模式跳过写 npy，不遗留孤儿文件
                    if not smoke:
                        np.save(signal_path, normalized)

                # 第三轮审查 3F-R3-16：删除 labels=None 死代码
                # 标签由第 4 步用当前 LabelExtractor 在线编码
                manifest.append({
                    "record_id": sample.record_id,
                    "source": sample.source,
                    "signal_file": signal_file,
                    "fs_original": float(sample.fs),
                    "fs_target": 500.0,
                    "duration_original": sample.duration,
                    "signal_shape": list(normalized.shape),
                    # 4C 复查（β）：导联序留痕——lead_names 非标准序且未重排
                    # （leads_matched=False）即"匹配失败保持原序"的告警记录。
                    # 注：权威 manifest 由 patient_level_split.py 从 .bak_old27
                    # 重写，不含这两个新字段（留痕仅在 preprocess 直接产出时有效）
                    "lead_names": list(sample.lead_names or []),
                    "leads_reordered": bool(leads_reordered),
                    "leads_matched": bool(leads_ok),
                    "age": sample.age if (sample.age is not None and not (
                        isinstance(sample.age, float) and
                        (np.isnan(sample.age) or np.isinf(sample.age)))) else None,
                    "sex": sample.sex,
                    "dx_codes": sample.dx_codes,
                    "has_labels": len(sample.dx_codes) > 0,
                })
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:
                logger.warning(f"处理失败 {key}: {e}")
                skipped += 1

        if args.max_records and count >= args.max_records:
            break

    pbar.close()
    # 2A 🟠 修复：加载失败原因显式汇总（旧版只报总数，原因不可见）
    loader_failures = getattr(loader, "failure_summary", lambda: {})()
    logger.info(f"处理完成: {len(manifest)} 条写入, 跳过 {skipped}, "
                f"导联重排 {reordered}, 不在旧划分 {unknown_split}")
    if loader_failures:
        logger.warning(f"加载失败原因分布: {loader_failures}")

    if args.max_records:
        logger.info("冒烟测试模式，不写 manifest。")
        # 快速统计
        logger.info(f"样本形状抽查: {[m['signal_shape'] for m in manifest[:3]]}")
        return

    # ---- 4. 按旧划分写 manifest（labels 用当前 LabelExtractor 在线编码）----
    # 第三轮审查 3B-DP-R3：旧版直接复制旧 manifest 的 labels（旧版码-名错配
    # LabelExtractor 的产物），重跑会把官方 27 类标签覆盖回错误标签。
    # 现改为在线用当前 LabelExtractor.encode(dx_codes) 生成。
    from src.data_pipeline.label_extractor import LabelExtractor as _LE
    _le = LabelExtractor()

    def _encode_labels(m):
        dx = m.get("dx_codes") or []
        # 4C-RED-1 修复：encode() 默认 multi_hot 返回 np.ndarray，
        # json 序列化必须 .tolist()（旧版全量重跑在写完 37k 条后才于
        # json.dump 处 TypeError 崩溃，前功尽弃）
        return _le.encode(dx).tolist() if dx else [0] * 27

    by_split = {"train": [], "val": [], "test": []}
    for m in manifest:
        m["labels"] = _encode_labels(m)
        by_split[split_map[(m["source"], m["record_id"])]].append(m)

    for split_name, files in by_split.items():
        mp = output_dir / f"{split_name}_manifest.json"
        with open(mp, "w", encoding="utf-8") as f:
            json.dump({"files": files, "count": len(files)}, f, indent=2)
        logger.info(f"  {split_name}: {len(files)} 条 → {mp}")

    # ---- 5. 复制报告文本（供后续多模态实验直接使用 processed_5k 目录）----
    reports_src = old_dir / "ptbxl_reports.json"
    if reports_src.exists():
        import shutil
        shutil.copy2(reports_src, output_dir / "ptbxl_reports.json")
        logger.info("已复制 ptbxl_reports.json")

    logger.info("=" * 50)
    logger.info(f"ECGFounder 协议预处理完成! 输出: {output_dir}")
    logger.info(f"Train/Val/Test: "
                f"{len(by_split['train'])}/{len(by_split['val'])}/{len(by_split['test'])}")


if __name__ == "__main__":
    main()
