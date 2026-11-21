# OSOT 标注质检与校准器

检查标注数据，对比机器初标和人工校对结果，辅助分析标注误差与一致性。

## 快速体验

```bash
python osot_qa_calibrator.py --demo --out-dir demo_qa_out
python osot_qa_calibrator.py --auto 初标.json --gold 人工终稿.json --out-dir qa_out
```

Windows 用户也可运行 `run_calibrator.bat`。仓库保留原版、v2 脚本和共用数据协议；启动脚本使用原版。
