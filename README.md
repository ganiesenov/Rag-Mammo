# Two-Stage Synthetic-to-Real Transfer Learning for Automated Mammography Report Generation

Official implementation of:  
**"Two-Stage Synthetic-to-Real Transfer Learning for Automated Mammography Report Generation Using Vision-Language Models"**  
*Gani Esen, Marat Nurtas — Scientific Reports (2025)*

## Overview

We propose a two-stage framework for mammography report generation that overcomes the scarcity of paired mammogram-report datasets:

- **Stage 1**: Synthetic pretraining — LLM-generated reports from BI-RADS annotations (VinDr-Mammo + CBIS-DDSM)
- **Stage 2**: Real-data fine-tuning — transfer to real radiologist reports (DMID)
- **RAG Module**: BI-RADS knowledge base (39 chunks, FAISS, MiniLM-L6-v2)
- **Model**: MedGemma-4B + LoRA (r=64, α=64), 4-bit NF4 quantization
- **Hardware**: Single NVIDIA RTX 5090 (32GB). Stage 1: ~3 min, Stage 2: ~37 min

## Results — DMID Test Set (n=52, real radiologist reports)

| Method | Type | BLEU-4 | ROUGE-L | METEOR | CIDEr | BERTScore |
|--------|------|--------|---------|--------|-------|-----------|
| LLaVA-1.6-7B | VLM, ZS | 0.001 | 0.109 | 0.188 | 0.005 | 0.810 |
| MedGemma-4B | VLM, ZS | 0.001 | 0.098 | 0.192 | 0.000 | 0.802 |
| Qwen2.5-VL-7B | VLM, ZS | 0.002 | 0.145 | 0.207 | 0.009 | 0.823 |
| Qwen3-VL-8B | VLM, ZS | 0.002 | 0.147 | 0.228 | 0.007 | — |
| MedCLIP+GPT2 | Pipeline, FT | 0.217 | 0.571 | 0.555 | — | 0.905 |
| CLIP+GPT2 | Pipeline, FT | 0.238 | 0.613 | 0.591 | — | 0.909 |
| AMRG (Sung et al.) | VLM, FT | 0.308 | 0.569 | 0.615 | 0.582 | — |
| DMID-only (LoRA) | VLM, FT | 0.203 | 0.601 | 0.597 | 0.455 | 0.909 |
| **Ours (two-stage)** | **VLM, 2S-FT** | **0.312** | **0.671** | **0.685** | **0.729** | **0.917** |

### vs AMRG (previous SOTA)
- **ROUGE-L**: +17.9% (0.671 vs 0.569)
- **METEOR**: +11.4% (0.685 vs 0.615)
- **CIDEr**: +25.3% (0.729 vs 0.582)

### Key Findings
- Hallucination rate: 73.1% (zero-shot) → 21.2% (two-stage)
- Data efficiency: 100 real examples sufficient to beat AMRG SOTA
- ~2× reduction in real data requirements
- LoRA ablation: r=64, α=64 optimal across 7 configurations

## LoRA Ablation

| r | α | Params | ROUGE-L | CIDEr |
|---|---|--------|---------|-------|
| 8 | 16 | 16.4M | 0.631 | 0.527 |
| 8 | 32 | 16.4M | 0.654 | 0.613 |
| 16 | 32 | 32.8M | 0.659 | 0.639 |
| 32 | 64 | 65.5M | 0.661 | 0.623 |
| **64** | **64** | **131.1M** | **0.671** | **0.729** |
| 64 | 128 | 131.1M | 0.666 | 0.687 |

## Architecture

![Architecture](paper/figures/fig_pipeline.png)

## Datasets

| Dataset | Cases | Images | Reports | Link |
|---------|-------|--------|---------|------|
| VinDr-Mammo | 5,000 | 20,000 | No | [Kaggle](https://www.kaggle.com/datasets/shantanughosh/vindr-mammogram-dataset-dicom-to-png) |
| CBIS-DDSM | 1,566 | — | No | [Kaggle](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset) |
| DMID | 225 | 510 | **Yes** | [Figshare](https://figshare.com/authors/Parita_Oza/17353984) |

## Setup

```bash
conda create -n rag_mammo python=3.13
conda activate rag_mammo
pip install torch transformers peft accelerate bitsandbytes
pip install sentence-transformers faiss-cpu rouge-score bert-score
pip install pycocoevalcap nltk
pip install python-telegram-bot  # for bot deployment
```

## Pipeline

```bash
# 1. Generate synthetic reports from structured annotations
python scripts/1_generate/generate_ollama.py

# 2. Build BI-RADS RAG knowledge base
python scripts/2_rag/build_rag.py

# 3. Stage 1: Synthetic pretraining
python scripts/3_finetune/finetune_multimodal.py

# 4. Stage 2: Real data fine-tuning (default r=16)
python scripts/3_finetune/finetune_dmid.py

# 5. LoRA ablation (r=8,16,32,64)
python scripts/3_finetune/lora_ablation.py

# 6. Evaluate all methods
python scripts/4_evaluate/phase2_full.py

# 7. Data efficiency analysis
python scripts/4_evaluate/data_efficiency.py
```

## Telegram Bot

Deploy the model as a Telegram bot for real-time mammography report generation:

```bash
export TELEGRAM_BOT_TOKEN="your_token_from_BotFather"
python bot/telegram_bot.py
```

Send a mammography image → get a structured BI-RADS report.

## Project Structure

```
new_article/
├── scripts/
│   ├── 1_generate/          # Synthetic report generation (Ollama/Llama-3.1-8B)
│   ├── 2_rag/               # BI-RADS RAG knowledge base
│   ├── 3_finetune/          # Stage 1 & 2 training, LoRA ablation
│   └── 4_evaluate/          # Evaluation scripts
├── bot/                     # Telegram bot
├── paper/                   # LaTeX paper + figures
├── results/                 # JSON results for all experiments
├── dmid/                    # DMID dataset (not included)
├── vindr/                   # VinDr-Mammo (not included)
└── cbis-ddsm/               # CBIS-DDSM (not included)
```

## Citation

```bibtex
@article{esen2025twostage,
  title={Two-Stage Synthetic-to-Real Transfer Learning for Automated Mammography Report Generation Using Vision-Language Models},
  author={Esen, Gani and Nurtas, Marat},
  journal={Scientific Reports},
  year={2025}
}
```

## License

This project is for research purposes only. Generated reports are not medical advice and must be reviewed by a qualified radiologist.