# Cross-Domain Generalization in Prompt Injection Detection

This repository presents cross-evaluation findings for prompt injection detection models, examining how detectors trained on one attack type generalize to unseen attack types.

## Research Overview

We trained two detection models using the PromptShield methodology (LoRA fine-tuning on Llama 3.1 8B) and evaluated their cross-domain generalization capabilities:

| Model | Training Data | Attack Type |
|-------|---------------|-------------|
| **PromptShield Model** | [hendzh/PromptShield](https://huggingface.co/datasets/hendzh/PromptShield) | Direct prompt injections |
| **LLMail Model** | [microsoft/llmail-inject-challenge](https://huggingface.co/datasets/microsoft/llmail-inject-challenge) | Indirect injections via email |

## Key Results

### Cross-Evaluation Matrix (TPR @ 1% FPR)

|  | → Direct Attacks | → Indirect Attacks |
|--|------------------|-------------------|
| **PromptShield Model** | **66.9%** (in-domain) | 19.3% (zero-shot) |
| **LLMail Model** | 0.0% (zero-shot) | **43.4%** (in-domain) |

### Performance Summary

| Metric | PromptShield Model | LLMail Model |
|--------|-------------------|--------------|
| In-Domain TPR @ 1% FPR | 66.9% | 43.4% |
| In-Domain TPR @ 0.1% FPR | 43.2% | 24.9% |
| In-Domain AUC | 0.970 | 0.871 |
| Zero-Shot TPR @ 1% FPR | 19.3% | 0.0% |
| Generalization Gap | 47.6 pts | 43.4 pts |

## Key Findings

### 1. Asymmetric Generalization

The PromptShield model (trained on direct attacks) retains partial detection capability on indirect attacks (19.3% TPR @ 1% FPR), while the LLMail model shows no transfer to direct attacks (0.0% TPR @ 1% FPR).

This suggests that **direct injection patterns may be more fundamental** and partially transferable, whereas indirect attack detection relies on domain-specific features (email structure, context manipulation) that don't generalize.

### 2. Domain-Specific Training Remains Critical

Average zero-shot performance (9.6% TPR @ 1% FPR) is dramatically lower than average in-domain performance (55.2%), demonstrating that:

- Single-domain training is insufficient for comprehensive protection
- Deployment requires either multi-domain training or ensemble approaches
- Attack-type-specific detectors may be necessary

### 3. Direct vs Indirect Attack Characteristics

| Characteristic | Direct Attacks | Indirect Attacks |
|----------------|----------------|------------------|
| **Injection Location** | User input | Retrieved content (emails, documents) |
| **Detection Cues** | Explicit override phrases | Subtle context manipulation |
| **Transferability** | Partial (19.3%) | None (0.0%) |

## Methodology

### Training Configuration

Both models use identical architecture and training setup:

```
Base Model:      Llama 3.1 8B
Fine-tuning:     LoRA (rank=16, alpha=32)
Precision:       4-bit quantization (QLoRA)
Learning Rate:   2e-4
Epochs:          3
Batch Size:      32 (effective)
```

### Evaluation Protocol

Following PromptShield methodology, we evaluate at deployment-critical FPR thresholds:

- **TPR @ 0.1% FPR**: Ultra-low false positive rate (1 false alarm per 1,000 requests)
- **TPR @ 1% FPR**: Standard deployment threshold
- **AUC**: Overall discriminative ability

Temperature scaling is applied for probability calibration before threshold-based evaluation.

## Detailed Results

### PromptShield Model Performance

```
In-Domain (Direct Attacks):
├── AUC:           0.970
├── TPR @ 1% FPR:  66.9%
├── TPR @ 0.5% FPR: 58.8%
├── TPR @ 0.1% FPR: 43.2%
└── TPR @ 0.05% FPR: 33.1%

Zero-Shot (Indirect Attacks):
├── TPR @ 1% FPR:  19.3%
└── Retention:     28.8%
```

### LLMail Model Performance

```
In-Domain (Indirect Attacks):
├── AUC:           0.871
├── TPR @ 1% FPR:  43.4%
├── TPR @ 0.5% FPR: 36.3%
├── TPR @ 0.1% FPR: 24.9%
└── TPR @ 0.05% FPR: 22.4%

Zero-Shot (Direct Attacks):
├── TPR @ 1% FPR:  0.0%
└── Retention:     0.0%
```

## Implications

### For Practitioners

1. **Don't assume generalization**: A detector trained on one attack type may fail completely on others
2. **Layer defenses**: Combine input screening with output monitoring
3. **Test across domains**: Evaluate on diverse attack types before deployment

### For Researchers

1. **Multi-domain training**: Investigate combined training on diverse attack types
2. **Transfer learning**: Develop methods to improve cross-domain generalization
3. **Attack taxonomy**: Better characterization of attack features that enable/prevent transfer

## Visualizations

### Cross-Evaluation Heatmap

```
                    Test Dataset
                    PromptShield    LLMail
              ┌─────────────────────────────┐
PromptShield  │     66.9%     │    19.3%    │
Model         │   (in-domain) │  (zero-shot)│
              ├───────────────┼─────────────┤
LLMail        │      0.0%     │    43.4%    │
Model         │  (zero-shot)  │  (in-domain)│
              └─────────────────────────────┘
```

### Generalization Gap Analysis

```
PromptShield: ████████████████████████████████░░░░░░░░░░░░░░  66.9% → 19.3% (-47.6 pts)
LLMail:       █████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░  43.4% →  0.0% (-43.4 pts)
              0%              25%             50%            75%           100%
```

## Repository Structure

```
├── notebooks/
│   ├── PromptShield_Training.ipynb      # Direct injection detector training
│   ├── LLMail_Training.ipynb            # Indirect injection detector training
│   └── Cross_Evaluation.ipynb           # Cross-domain evaluation
├── results/
│   ├── promptshield_results.json        # PromptShield model metrics
│   ├── llmail_results.json              # LLMail model metrics
│   └── cross_eval_results.json          # Cross-evaluation metrics
└── figures/
    ├── cross_eval_heatmap.png
    └── generalization_comparison.png
```

## Citation

If you use these findings in your research, please cite:

```bibtex
@misc{promptinjection_crosseval_2024,
  title={Cross-Domain Generalization in Prompt Injection Detection},
  author={[Your Name]},
  year={2024},
  note={Undergraduate Thesis Research}
}
```

## References

1. **PromptShield**: Hend Alzahrani et al. "PromptShield: A Unified Framework for Prompt Injection Detection" (UC Berkeley, David Wagner Research Group)
2. **LLMail-Inject**: Microsoft Security Research, SaTML 2025 Competition Dataset
3. **LoRA**: Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models" (2021)

## License

MIT License - See [LICENSE](LICENSE) for details.

---

**Note**: This research was conducted as part of an undergraduate thesis on AI security, focusing on prompt injection detection systems and their deployment viability at extremely low false positive rates.
