# 模型比较可视化工具使用说明

本文档介绍如何使用`run_visualization.sh`脚本比较不同模型在ScreenSpot任务上的表现。

## 前提条件

- 确保已安装Python和相关依赖库（OpenCV、NumPy等）
- 确保`visualize_model_comparison.py`脚本在当前目录
- 确保模型评估结果文件已生成

## 脚本功能

`run_visualization.sh`脚本可以：

1. 比较训练后模型和baseline模型在点击任务上的性能差异
2. 使用绿色框标注ground_truth，红色点标注训练后模型结果，橙色点标注baseline模型结果
3. 显示各坐标值和正确/错误状态
4. 支持处理桌面、移动和网页三个平台的数据
5. 支持限制处理样本数量、过滤正确/错误样本等

## 快速开始

### 基本用法

```bash
# 使用默认参数运行
./run_visualization.sh

# 只处理桌面平台数据
./run_visualization.sh --platform desktop

# 限制处理10个样本
./run_visualization.sh --limit 10

# 只处理训练后模型正确的样本
./run_visualization.sh --correct-only true
```

### 高级用法

```bash
# 组合使用多个参数
./run_visualization.sh --platform mobile --limit 50 --output-dir vis_results/mobile_only

# 处理所有样本（设置limit为0）
./run_visualization.sh --platform all --limit 0

# 查看帮助信息
./run_visualization.sh --help
```

## 命令行参数说明

| 参数 | 简写 | 描述 | 默认值 |
|------|------|------|--------|
| `--help` | `-h` | 显示帮助信息 | - |
| `--platform` | `-p` | 指定处理平台 (desktop, mobile, web, all) | all |
| `--limit` | `-l` | 限制处理样本数量（0表示所有样本） | 20 |
| `--correct-only` | `-c` | 仅处理训练后模型正确的样本 | false |
| `--incorrect-only` | `-i` | 仅处理训练后模型错误的样本 | false |
| `--output-dir` | `-o` | 指定输出目录 | vis_results/model_comparison |

## 配置参数

如需修改其他参数（如点的半径、线条粗细等），可以直接编辑脚本开头的配置部分：

```bash
# 配置参数（请根据需要修改）
# 基础路径
BASE_PATH="/c22940/zy/code/VLM-R1"
# 训练后模型结果路径
TUNED_MODEL_PATH="logs/Qwen2.5-VL-7B-GRPO-ScreenSpot-Desktop-Click/qwen2.5-vl-7b-grpo-screenspot-desktop-click-checkpoint-334"
# Baseline模型结果路径
BASELINE_MODEL_PATH="logs/Qwen2.5-VL-7B-GRPO-ScreenSpot-Desktop-Click/qwen2_5vl_7b_instruct_baseline"
# 点半径大小
POINT_RADIUS=5
# 边界框线条粗细
THICKNESS=3
# 边界框透明度
ALPHA=0.3
```

## 输出结果

脚本运行后，会在指定的输出目录（默认为`vis_results/model_comparison`）下生成按平台分类的子目录：

- `desktop/`：桌面平台的比较结果
- `mobile/`：移动平台的比较结果
- `web/`：网页平台的比较结果

每个图像文件名格式为：`<图像名>_<结果类型>.png`，其中`<结果类型>`可能是：

- `both_correct`：两个模型都正确
- `both_wrong`：两个模型都错误
- `tuned_better`：训练后模型正确而Baseline错误
- `baseline_better`：Baseline模型正确而训练后模型错误

## 常见问题

**问题1: 脚本执行权限不足**
```bash
chmod +x run_visualization.sh
```

**问题2: 找不到Python脚本**
确保`visualize_model_comparison.py`文件在当前目录，或修改`BASE_PATH`指向正确位置。

**问题3: 找不到结果文件**
检查`TUNED_MODEL_PATH`和`BASELINE_MODEL_PATH`配置是否正确。

## 示例结果

可视化结果示例：
- 绿色框：ground_truth边界框，包含[x1,y1,x2,y2]坐标
- 红色点：训练后模型点击位置，包含[x,y]坐标和正确/错误标记
- 橙色点：baseline模型点击位置，包含[x,y]坐标和正确/错误标记

## 进阶定制

如需进一步定制可视化效果（如修改颜色、标记方式等），请直接修改`visualize_model_comparison.py`文件。 