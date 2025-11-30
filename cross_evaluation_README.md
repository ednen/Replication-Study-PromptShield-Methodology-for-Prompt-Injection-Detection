 Cross-Domain Generalization in Prompt Injection Detection

This repository presents cross-evaluation findings for prompt injection detection models, examining how detectors trained on one attack type generalize to unseen attack types.

 Research Overview

We trained two detection models using the PromptShield methodology (LoRA fine-tuning on Llama 3.1 8B) and evaluated their cross-domain generalization capabilities:

| Model | Training Data | Attack Type |
|-------|---------------|-------------|
| PromptShield Model | [hendzh/PromptShield](https://huggingface.co/datasets/hendzh/PromptShield) | Direct prompt injections |
| LLMail Model | [microsoft/llmail-inject-challenge](https://huggingface.co/datasets/microsoft/llmail-inject-challenge) | Indirect injections via email |

 Key Results

 Cross-Evaluation Matrix (TPR @ 1% FPR)

|  |  Direct Attacks |  Indirect Attacks |
|--|------------------|-------------------|
| PromptShield Model | 66.4% (in-domain) | 68.6% (zero-shot) |
| **LLMail Model** | 0.0% (zero-shot) | 43.4% (in-domain) |

 Performance Summary

| Metric | PromptShield Model | LLMail Model |
|--------|-------------------|--------------|
| In-Domain TPR @ 1% FPR | 66.4% | 43.4% |
| In-Domain TPR @ 0.1% FPR | 43.2% | 24.9% |
| In-Domain AUC | 0.970 | 0.871 |
| Zero-Shot TPR @ 1% FPR | 68.6% | 0.0% |
| Generalization Gap | -2.2 pts | +43.4 pts |

 Key Findings

 1. PromptShield Shows Robust Cross-Domain Transfer

The PromptShield model (trained on direct attacks) achieves **higher performance on indirect attacks** (68.6%) than on its own training domain (66.4%). This remarkable finding suggests:

- Learned generalizable injection patterns that transcend attack delivery mechanisms
- Direct injection training captures fundamental adversarial features useful across domains
- The model's detection capabilities are not overfitted to direct attack formats

 2. Asymmetric Generalization

While PromptShield generalizes excellently to indirect attacks, the reverse is not true:

| Direction | TPR @ 1% FPR | Interpretation |
|-----------|--------------|----------------|
| Direct → Indirect | 68.6% | **Excellent transfer** |
| Indirect → Direct | 0.0% | **No transfer** |

This asymmetry suggests that **direct injection patterns are more fundamental** and transferable, whereas indirect attack detection relies on domain-specific features (email structure, context manipulation) that don't generalize.

 3. Implications for Defense Strategy

- Direct injection training provides broad protection: A model trained on direct attacks offers meaningful defense against indirect attacks
- Indirect injection training is specialized: Models trained only on indirect attacks fail completely on direct attacks
- Recommended approach: Train on direct injections as a baseline, with optional domain-specific fine-tuning

 Methodology

 Training Configuration

Both models use identical architecture and training setup:

```
Base Model:      Llama 3.1 8B
Fine-tuning:     LoRA (rank=16, alpha=32)
Precision:       4-bit quantization (QLoRA)
Learning Rate:   2e-4
Epochs:          3
Batch Size:      32 (effective)
```

 Evaluation Protocol

Following PromptShield methodology, we evaluate at deployment-critical FPR thresholds:

- TPR @ 0.1% FPR
- TPR @ 1% FPR
- AUC

Temperature scaling is applied for probability calibration before threshold-based evaluation.

 Detailed Results

 PromptShield Model Performance

```
In-Domain (Direct Attacks):
├── AUC:           0.970
├── TPR @ 1% FPR:  66.4%
├── TPR @ 0.5% FPR: 58.8%
├── TPR @ 0.1% FPR: 43.2%
└── TPR @ 0.05% FPR: 33.1%

Zero-Shot (Indirect Attacks):
├── TPR @ 1% FPR:  68.6%
└── Retention:     103.3% (exceeds in-domain!)
```

 LLMail Model Performance

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

 Visualizations

 Cross-Evaluation Heatmap

```
                    Test Dataset
                    PromptShield    LLMail
              ┌─────────────────────────────┐
PromptShield  │     66.4%     │    68.6%    │
Model         │   (in-domain) │  (zero-shot)│
              ├───────────────┼─────────────┤
LLMail        │      0.0%     │    43.4%    │
Model         │  (zero-shot)  │  (in-domain)│
              └─────────────────────────────┘
```

 Generalization Comparison

```
PromptShield: ██████████████████████████████████  66.4% → 68.6% (+2.2 pts) ✓
LLMail:       █████████████████████░░░░░░░░░░░░░  43.4% →  0.0% (-43.4 pts) ✗
              0%              25%             50%            75%           100%
```

 Implications

 For Practitioners

1. Prioritize direct injection training: Models trained on direct attacks generalize well to indirect attacks
2. Don't rely solely on indirect attack training: Such models fail on direct attacks
3. Layer defenses: Combine a direct-injection-trained detector with domain-specific monitoring

 For Researchers

1. Investigate what makes direct patterns transferable: Understanding this could improve detector design
2. Explore multi-domain training: Can combined training achieve best of both worlds?
3. Attack feature analysis: Characterize which features enable/prevent cross-domain transfer

 Repository Structure

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

 Citation

If you use these findings in your research, please cite:

```bibtex
@misc{promptinjection_crosseval_2025,
  title={Cross-Domain Generalization in Prompt Injection Detection},
  author={Ozan Bülen},
  year={2025},
  note={Undergraduate Thesis Research}
}
```

 References

1. PromptShield: Hend Alzahrani et al. "PromptShield: A Unified Framework for Prompt Injection Detection" (UC Berkeley, David Wagner Research Group)
2. LLMail-Inject: Microsoft Security Research, SaTML 2025 Competition Dataset
3. LoRA: Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
4. Claude 4.5 Sonnet

 License

MIT License - See [LICENSE](LICENSE) for details.

---

Note: This research was conducted as part of an undergraduate thesis on AI security, focusing on prompt injection detection systems and their deployment viability at extremely low false positive rates.
Note on LLMail Direct transfer: The 0% result may be partially 
attributed to input format mismatch (LLMail trained on email format, 
tested on raw prompts). The model may have some detection capability 
that falls below the 1% FPR threshold.
