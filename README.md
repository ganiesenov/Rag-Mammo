# RAG-Mammo: RAG-Augmented VLM for Mammography Report Generation

Official implementation of the paper:
**"RAG-Augmented Vision-Language Model for Automated Mammography Report Generation Without Paired Training Data"**

## Overview

We propose a framework for automated mammography report generation that requires **no real radiology reports** during training. Our approach combines:

1. **Label-to-Report Synthesis** — converts BI-RADS annotations to synthetic reports using Llama-3.1-8B (Ollama)
2. **RAG Module** — BI-RADS knowledge base indexed with FAISS for clinical grounding
3. **Multimodal Fine-tuning** — MedGemma-4B with LoRA on synthetic image-report pairs

## Results

| Method | ROUGE-1 | BERTScore |
|--------|---------|-----------|
| LLaVA-1.6-7B (zero-shot) | 0.221 | 0.818 |
| Qwen2.5-VL-7B (zero-shot) | 0.412 | 0.860 |
| MedGemma-4B (zero-shot) | 0.365 | 0.843 |
| **Ours (Multimodal FT)** | **0.612** | **0.919** |

## Datasets

- [VinDr-Mammo](https://www.kaggle.com/datasets/shantanughosh/vindr-mammogram-dataset-dicom-to-png)
- [CBIS-DDSM](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)
- [DMID](https://figshare.com/authors/Parita_Oza/17353984)

## Setup
```bash
conda create -n rag_mammo python=3.10
conda activate rag_mammo
pip install transformers peft accelerate bitsandbytes trl datasets
pip install sentence-transformers faiss-cpu rouge-score bert-score
pip install ollama  # for report synthesis
```

## Pipeline
```bash
# 1. Generate synthetic reports
python scripts/1_generate/generate_ollama.py

# 2. Build RAG knowledge base
python scripts/2_rag/build_rag.py

# 3. Fine-tune MedGemma-4B
python scripts/3_finetune/finetune_multimodal.py

# 4. Evaluate
python scripts/4_evaluate/evaluate_multimodal.py
```

## Citation
```bibtex
@article{jumagali2025ragmammo,
  title={RAG-Augmented Vision-Language Model for Automated Mammography Report Generation Without Paired Training Data},
  author={Jumagali, Gani},
  journal={Scientific Reports},
  year={2025}
}
```
