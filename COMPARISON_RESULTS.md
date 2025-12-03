# PromptShield vs Meta Prompt Guard 2: Comparative Evaluation

A comprehensive comparison of prompt injection detection models at deployment-critical thresholds.

## Results Summary

| Benchmark | Prompt Guard 2 | PromptShield (Ours) | Winner |
|-----------|----------------|---------------------|--------|
| Mixed Attacks (TPR@1%FPR) | 16.5% | 66.7% | PromptShield |
| Indirect Attacks (TPR@1%FPR) | 87.6% | 71.1% | Prompt Guard 2 |
| Over-Defense (Accuracy) | 98.5% | 100.0% | PromptShield |

## Key Findings

1. PromptShield achieves 4x higher detection on mixed attacks - 66.7% vs 16.5% TPR at 1% FPR, indicating superior performance on the standard PromptShield benchmark.

2. Prompt Guard 2 excels at indirect attacks - 87.6% vs 71.1% on email-based injections (LLMail), suggesting optimization for embedded/indirect threat vectors.

3. PromptShield has zero false positives - Perfect 100% accuracy on NotInject benchmark (0/339 false alarms) vs Prompt Guard 2's 5 false positives.

## Methodology

### Evaluation Datasets

| Dataset | Type | Samples | Description |
|---------|------|---------|-------------|
| [PromptShield](https://huggingface.co/datasets/hendzh/PromptShield) | Mixed Attacks | 23,516 | Standard prompt injection benchmark (test split) |
| [LLMail-Inject](https://huggingface.co/datasets/microsoft/llmail-inject-challenge) | Indirect Attacks | 6,000 | Email-based injection attacks |
| [NotInject](https://huggingface.co/datasets/leolee99/NotInject) | Over-Defense | 339 | Benign prompts with trigger words |

### Models

| Model | Parameters | Architecture |
|-------|------------|--------------|
| [Meta Prompt Guard 2](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M) | 86M | mDeBERTa-based classifier |
| PromptShield (Ours) | 8B (LoRA) | Llama 3.1 8B + LoRA fine-tuning |

### Evaluation Protocol

- Metric: True Positive Rate (TPR) at 1% False Positive Rate (FPR)
- Calibration: Temperature scaling (T=5.89) applied to PromptShield
- Thresholds: Calibrated independently on each model's mixed attack performance

## Detailed Results

### Mixed Attacks (PromptShield Benchmark)

| Model | TPR@1%FPR | TPR@0.1%FPR | AUC |
|-------|-----------|-------------|-----|
| Prompt Guard 2 | 16.5% | - | 0.853 |
| PromptShield | 66.7% | 43.3% | 0.966 |

### Indirect Attacks (LLMail Email Injections)

| Model | TPR@1%FPR | AUC |
|-------|-----------|-----|
| Prompt Guard 2 | 87.6% | 0.995 |
| PromptShield | 71.1% | 0.929 |

### Over-Defense (NotInject)

| Model | Accuracy | False Positives |
|-------|----------|-----------------|
| Prompt Guard 2 | 98.5% | 5/339 |
| PromptShield | 100.0% | 0/339 |

## Interpretation

The results reveal complementary strengths between the two approaches:

- PromptShield excels at detecting mixed/direct prompt injections while maintaining zero false positives on benign inputs with trigger words.

- Prompt Guard 2 demonstrates superior generalization to indirect attack vectors (email-based injections) despite weaker performance on the standard benchmark.

This suggests different optimization targets: PromptShield for precision on direct attacks, Prompt Guard 2 for broader indirect threat coverage.

## Citation

```bibtex
@misc{promptshield_comparison_2025,
  title={PromptShield Replicate vs Meta Prompt Guard 2: Comparative Evaluation},
  author={Ozan Bülen},
  year={2025},
  url={(https://github.com/ednen/Replication-Study-PromptShield-Methodology-for-Prompt-Injection-Detection/tree/cross-eval)}
}
```

## License

MIT License
