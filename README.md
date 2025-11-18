# PromptShield Replication: Prompt Injection Detection with Llama 3.1

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview

This repository contains an implementation replicating the PromptShield methodology for detecting prompt injection attacks on Large Language Models (LLMs). The work validates PromptShield's approach using Llama 3.1 8B with Low-Rank Adaptation (LoRA) fine-tuning and demonstrates deployment-viable performance on the PromptShield benchmark dataset.Model itself can be reached from https://huggingface.co/Ednen/promptshieldreplicate/tree/main.

## Motivation

As Large Language Models are increasingly integrated into production applications, they become vulnerable to prompt injection attacks where malicious instructions are embedded in user inputs to hijack model behavior. Existing detection methods have focused primarily on overall accuracy metrics, which can be misleading for security applications where false positive rates must be minimized to maintain system usability.

PromptShield introduced a critical innovation by evaluating detection performance at low false positive rate (FPR) thresholds, directly addressing the deployment constraints of security systems. This replication validates their methodology and provides an open-source implementation for the research community.

## Research Questions

1. Can PromptShield's methodology be successfully replicated using alternative model architectures?
2. What performance can be achieved at deployment-critical low FPR thresholds (0.05% - 1%)?
3. How does temperature scaling affect probability calibration and deployment viability?
4. What are the practical considerations for production deployment of such systems?

## Methodology

### Model Architecture

The implementation uses Llama 3.1 8B, an instruction-tuned language model, fine-tuned for binary sequence classification (benign vs. prompt injection). Key technical decisions:

- Base Model: meta-llama/Llama-3.1-8B
- Training Method: Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning
- Quantization: 4-bit QLoRA for efficient training on single GPU
- LoRA Configuration: rank=16, alpha=32, target modules=[q_proj, k_proj, v_proj, o_proj]

### Dataset

Training and evaluation use the PromptShield benchmark dataset (hendzh/PromptShield on HuggingFace), which contains approximately 19,000 training examples and 23,500 test examples. The dataset includes both conversational data and application-structured data to reflect realistic deployment scenarios.

### Training Configuration

- Optimizer: AdamW
- Learning Rate: 2e-4
- Batch Size: 32 (effective, via gradient accumulation)
- Epochs: 3
- Max Sequence Length: 512 tokens
- Precision: bfloat16 mixed precision training
- Hardware: Single A100 GPU

### Evaluation Methodology

Following PromptShield's approach, the primary evaluation focuses on True Positive Rate (TPR) at specific low False Positive Rate (FPR) thresholds:

- 0.05% FPR
- 0.10% FPR
- 0.50% FPR
- 1.00% FPR

This evaluation methodology directly addresses deployment constraints where false alarms must be minimized to maintain system usability. Traditional metrics (AUC, overall accuracy) are also reported for comparison with prior work.

### Temperature Scaling

Post-training probability calibration using temperature scaling is implemented following Guo et al. (2017). A temperature parameter T is optimized on the validation set to minimize negative log-likelihood, then applied to scale model outputs during inference:

```
calibrated_probabilities = softmax(logits / T)
```

This calibration is critical for setting meaningful decision thresholds and achieving practical FPR targets.

## Results

### Low FPR Performance

| FPR Level | TPR (This Work) | TPR (PromptShield) | Required Threshold |
|-----------|-----------------|--------------------|--------------------|
| 0.05%     | 35.8%           | ~45-50%*           | 0.992              |
| 0.10%     | 43.3%           | ~55-60%*           | 0.988              |
| 0.50%     | 58.4%           | ~65%*              | 0.966              |
| 1.00%     | 66.3%           | 73.82%             | 0.947              |

*Estimated from PromptShield paper figures

### Overall Metrics

- Accuracy: 96.0%
- Precision: 92.6%
- Recall: 93.1%
- F1 Score: 92.8%
- AUC: 0.966

### Comparison to Baselines

This work achieves 66.3% TPR at 1% FPR, compared to prior work (PromptGuard, DeBERTa-based models) which achieved less than 20% TPR at the same FPR threshold. The results validate PromptShield's methodology while demonstrating deployment viability.

## Analysis

### Performance Relative to PromptShield

The 7.5 percentage point gap at 1% FPR (66.3% vs. 73.82%) is attributable to several factors:

1. Model Architecture Differences: Llama 3.1 8B (decoder-only, autoregressive) vs. FLAN-T5-large (encoder-decoder)
2. Quantization: 4-bit QLoRA vs. likely full-precision training
3. Training Details: Different hyperparameter choices, number of epochs, optimizer settings
4. Dataset Sampling: Possible variations in random seed and train/test splits

Despite these differences, the results demonstrate successful replication of the core methodology and validate the low FPR evaluation approach.

### Deployment Viability

At the recommended 0.1% FPR operating point:

- Detection Rate: 43.3% of attacks
- False Alarm Rate: 1 per 1,000 legitimate requests
- For 1M daily requests: ~1,000 false alarms (40 per hour), 4,330 attacks detected

This represents a practical deployment point where false alarms are manageable with human review queues while providing meaningful security improvement. The system would operate as one layer in a defense-in-depth architecture alongside input validation, rate limiting, and output monitoring.

### Impact of Temperature Scaling

Temperature scaling enables practical threshold selection by calibrating model confidence to match true probabilities. Without calibration, achieving 0.1% FPR would require extremely high confidence thresholds (potentially >99.5%), yielding minimal detection. With calibration, a 98.8% threshold achieves the target FPR with 43.3% TPR.

## Technical Implementation

### Model Training

The training pipeline implements:

- Proper Llama 3.1 tokenization with pad token configuration
- LoRA parameter-efficient fine-tuning
- Mixed precision training (bfloat16) optimized for A100 GPUs
- Gradient accumulation for effective larger batch sizes
- Comprehensive evaluation at multiple FPR thresholds

### Probability Calibration

Temperature scaling is implemented via scipy.optimize to find the optimal temperature parameter that minimizes negative log-likelihood on the validation set. The calibrated model produces well-calibrated probabilities suitable for threshold-based decision making.

### Evaluation Framework

The evaluation includes:

1. Standard classification metrics (accuracy, precision, recall, F1, AUC)
2. Low FPR evaluation at 0.05%, 0.1%, 0.5%, and 1% FPR
3. ROC curve visualization with focus on low FPR region
4. Three-tier adversarial testing framework

## Repository Structure

```
.
├── notebooks/
│   └── PromptShield_Llama3_1_Training.ipynb    # Complete training pipeline
├── results/
│   ├── low_fpr_results.json                     # Low FPR evaluation metrics
│   ├── test_results.json                        # Standard evaluation metrics
│   └── low_fpr_roc_curve.png                   # Visualization
└── README.md
```

## Reproduction Instructions

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (A100 recommended, 40GB VRAM)
- HuggingFace account with access to Llama 3.1 models

### Dependencies

```bash
pip install transformers datasets accelerate bitsandbytes scipy scikit-learn peft trl
```

### Running the Notebook

1. Obtain HuggingFace access token for Llama 3.1 models
2. Open the training notebook in Google Colab or Jupyter
3. Configure HuggingFace authentication
4. Execute cells sequentially
5. Results will be saved to Google Drive (Colab) or local directory

### Key Hyperparameters

The following hyperparameters can be adjusted in the training configuration:

- LoRA rank (r): Controls adaptation capacity vs. efficiency trade-off
- Learning rate: Recommended range 1e-4 to 3e-4 for LoRA
- Epochs: Typically 2-5 epochs sufficient for convergence
- Batch size: Scale according to available GPU memory

## Limitations and Future Work

### Current Limitations

1. Language Coverage: Primarily trained and evaluated on English text
2. Attack Evolution: Model trained on known attack patterns; new attack types may evade detection
3. Adversarial Robustness: Not explicitly trained against adaptive adversaries
4. False Negative Rate: 56.7% of attacks missed at 0.1% FPR deployment point
5. Computational Cost: 10-20ms inference latency per request may be prohibitive for high-throughput systems

### Future Directions

1. Multi-Model Comparison: Systematic evaluation of different model sizes and architectures
2. Ablation Studies: Quantifying impact of individual components (temperature scaling, LoRA rank, quantization)
3. Extended Evaluation: Testing on additional datasets and real-world attack samples
4. Adversarial Training: Incorporating adversarial examples during training
5. Model Distillation: Creating smaller, faster models for edge deployment
6. Multilingual Evaluation: Extending to non-English languages
7. Ensemble Methods: Combining multiple models for improved robustness

## Key Contributions

1. Independent validation of PromptShield's low FPR evaluation methodology
2. Demonstration that the approach generalizes to alternative model architectures (Llama vs. FLAN-T5)
3. Deployment-focused analysis with practical threshold recommendations
4. Open-source implementation enabling further research
5. Comprehensive documentation of replication process and results

## Citations

If you use this work, please cite:

```bibtex
@misc{promptshield_replication_2025,
  title={PromptShield Replication: Prompt Injection Detection with Llama 3.1},
  author={[Ozan Bülen]},
  year={2025},
  howpublished={https://github.com/ednen/Replication-Study-PromptShield-Methodology-for-Prompt-Injection-Detection}
}
```

Original PromptShield paper:

```bibtex
@inproceedings{jacob2025promptshield,
  title={PromptShield: Deployable Detection for Prompt Injection Attacks},
  author={Jacob, Dennis and Alzahrani, Hend and Hu, Zhanhao and Alomair, Basel and Wagner, David},
  booktitle={Proceedings of the Fifteenth ACM Conference on Data and Application Security and Privacy},
  year={2025}
}
```

## References

1. Jacob, D., Alzahrani, H., Hu, Z., Alomair, B., & Wagner, D. (2025). PromptShield: Deployable Detection for Prompt Injection Attacks. ACM Conference on Data and Application Security and Privacy (CODASPY).

2. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv preprint arXiv:2106.09685.

3. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. International Conference on Machine Learning (ICML).

4. Touvron, H., et al. (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models. arXiv preprint arXiv:2307.09288.

5. Claude 4.5 Sonnet.

## License

This research implementation is provided for academic and research purposes. The Llama 3.1 model is subject to Meta's license agreement. The PromptShield dataset is subject to its original license terms.

## Contact

For questions, issues, or collaboration opportunities, please open an issue on this repository or contact [your contact information].

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Usage Rights

The MIT License permits:
- Commercial use
- Modification
- Distribution
- Private use

Requirements:
- Include copyright notice and license text in copies
- Provide attribution when using this work

Note: The Llama 3.1 model weights are subject to Meta's license agreement, and the PromptShield dataset has its own license terms. Users must comply with all applicable licenses.

## Acknowledgments

This work builds upon the PromptShield methodology developed by David Wagner's research group at UC Berkeley. We acknowledge their contribution to advancing LLM security research and making their dataset publicly available.
