# Model Editing Project

本项目主要用于探索和应用大语言模型（LLM）的知识编辑技术，包含常用的模型编辑方法如 ROME 和 MEMIT。通过本项目你可以测试模型在编辑前后的表现，支持数据的批量处理以及修改效果的评估。

## 项目结构
- `baseline.py`：测试基线，在不加任何知识编辑前的模型推理测试。
- `download_hf_data.py` / `prepare_batch_data.py`：用于从 Hugging Face 下载所需数据集或模型并构建我们需要的批量评测或编辑数据源 (`data.json`, `batch_data.json`)。
- `edit_rome.py` / `edit_memit.py`：核心编辑脚本，分别应用 ROME 和 MEMIT 算法对模型进行知识编辑。
- `evaluate.py`：评估脚本，对编辑后的模型进行打分和性能评估。
- `install.py` / `requirements.txt`：项目运行环境的安装与依赖。
- `hparams/`：存放模型编辑算法需要的超参数配置。

## 环境准备

首先安装所需的 Python 依赖包：

```bash
pip install -r requirements.txt
```

如果项目还需要一些其他的特定依赖初始化，请运行：

```bash
python install.py
```

## 运行步骤

### 1. 数据准备
首先获取必要的数据，并准备好我们要修改的知识事实（facts）：

```bash
python download_hf_data.py
python prepare_batch_data.py
```

### 2. 基线测试 (Baseline)
在不对模型进行修改的情况下，测试预训练模型对其知识的掌握情况：

```bash
python baseline.py
```

### 3. 模型知识编辑
根据你想要使用的编辑方法，运行对应的脚本。以 ROME 为例：

```bash
python edit_rome.py
```
*(或者使用 MEMIT: `python edit_memit.py`)*

编辑后的模型通常会保存在本地（例如 `edited_gpt2_rome/` 目录下）。

### 4. 评估结果
最后，使用评估脚本来测试编辑的成功率和模型在其他数据上的泛化/退化情况：

```bash
python evaluate.py
```

## 日志与产出
程序的运行日志存放在 `logs/` 目录下，编辑后的模型文件及衍生统计数据可在 `edited_gpt2_rome/` 或 `data/stats/` 等相关目录中找到。
