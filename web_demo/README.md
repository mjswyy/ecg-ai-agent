# ECG AI Agent — 情境条件化双层诊断演示

## 运行

```powershell
cd C:\Users\llyun\Desktop\ecg资料\ecg-ai-agent
python web_demo/app.py
```

浏览器打开 http://127.0.0.1:7860

## 前置条件

| 依赖 | 说明 |
|------|------|
| 模型权重 | `checkpoints/ECGFounder/12_lead_ECGFounder.pth` + `outputs/ecgfounder_mlp/mlp_head.pt` |
| 知识库 | `data/knowledge/kb/classes.json`（先跑 `compile_knowledge.py` 生成） |
| 数据 | `data/physionet2020/processed_5k/`（测试集选择用） |
| LLM（可选） | 根目录 `.env` 写入 `DEEPSEEK_API_KEY=sk-xxx`；无 key 时自动降级为"无 LLM 模式"（工具链与发现层仍可用） |

## 功能

- **输入**：测试集记录（3,754 条，患者级划分）或上传 WFDB 文件（.hea+.mat）+ 患者情境（年龄/性别/场景/主诉/病史）
- **输出**（四个标签页）：
  1. 波形与发现：12 导联图（Ⅱ 导联 R 峰标注）+ 工具链数值 + ECGFounder 27 类发现（0.9456）
  2. 双层诊断报告：发现层（情境无关）+ 风险层/建议层（情境条件化，DeepSeek 生成）
  3. 知识库依据：分级可信医学参考资料（[来源,等级] 引用）
  4. 参考答案（真值）：**仅选中测试集记录时**显示 PhysioNet 官方标注（`dx_codes` + 27 维
     `labels`），并与分类器阳性逐类对照（命中 / 漏报 / 额外阳性，含逐类概率与 Youden 阈值）；
     上传文件时明确提示"真实使用场景无真值"

## 设计原则

同一份心电图在不同情境下：**发现不变，风险与建议随情境调整**。
LLM 仅见特征摘要与知识库片段，不见原始波形（防幻觉设计）。

真值页签**只写界面**：真值不进入发现层、不进入 LLM prompt——发现层内容必须完全来自
分类器阳性（论文 §4.4 的 570/570 一致性口径）。因此新增该页签不改变任何评测数字，
三臂评测无需重跑。真值页签中的「额外阳性」≠ 误报：PhysioNet 标注非穷尽。

