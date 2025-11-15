# PromptShield Replication: Executive Summary

## Project Overview

This project replicates the PromptShield methodology for detecting prompt injection attacks on Large Language Models. Using Llama 3.1 8B with LoRA fine-tuning, the implementation achieves 66.3% true positive rate at 1% false positive rate on the PromptShield benchmark, validating the original work's approach while demonstrating practical deployment viability.

## Problem Statement

Large Language Models integrated into applications are vulnerable to prompt injection attacks, where malicious instructions embedded in inputs hijack model behavior. Existing detection methods focus on overall accuracy rather than deployment-critical metrics, specifically performance at low false positive rates where security systems must operate to remain usable.

## Approach

The implementation fine-tunes Llama 3.1 8B using Low-Rank Adaptation (LoRA) on the PromptShield benchmark dataset containing approximately 19,000 training examples. Following PromptShield's methodology, evaluation emphasizes True Positive Rate (TPR) at specific low False Positive Rate (FPR) thresholds: 0.05%, 0.1%, 0.5%, and 1%. Temperature scaling calibrates output probabilities to enable practical threshold selection.

## Key Results

Performance at deployment-critical FPR thresholds:
- 0.10% FPR: 43.3% TPR (recommended production setting)
- 1.00% FPR: 66.3% TPR (7.5 percentage points below PromptShield's 73.82%)

Overall metrics:
- Accuracy: 96.0%
- AUC: 0.966
- F1 Score: 92.8%

The results significantly outperform prior work (less than 20% TPR at 1% FPR) and validate PromptShield's methodology using an alternative model architecture.

## Deployment Analysis

At the recommended 0.1% FPR operating point with 1 million daily requests:
- False alarms: 1,000 per day (40 per hour)
- Attack detection: 4,330 out of 10,000 attacks (43.3%)
- Missed attacks: 5,670 (56.7%)

This represents a practical deployment configuration where false alarms are manageable with human review queues while providing meaningful security improvement as part of a defense-in-depth strategy.

## Technical Implementation

Core components:
- Model: Llama 3.1 8B with 4-bit QLoRA quantization
- Training: 3 epochs, learning rate 2e-4, effective batch size 32
- LoRA: rank 16, alpha 32, targeting attention projection layers
- Calibration: Temperature scaling on validation set
- Hardware: Single A100 GPU, bfloat16 mixed precision

## Contributions

1. Independent validation of PromptShield's low FPR evaluation methodology
2. Demonstration of approach generalization to alternative architectures
3. Deployment-focused analysis with practical operating point recommendations
4. Open-source implementation for research community
5. Comprehensive documentation of replication process

## Limitations

The implementation has several acknowledged limitations:
- Primary focus on English language text
- Vulnerability to novel attack patterns not present in training data
- No explicit adversarial robustness training
- 56.7% false negative rate at recommended operating point
- 10-20ms inference latency may constrain high-throughput applications

## Significance

This work validates that machine learning-based prompt injection detection is viable for production deployment. The 43.3% detection rate at 0.1% FPR represents a practical operating point for security systems, significantly improving upon prior approaches while maintaining acceptable false alarm rates. The replication confirms PromptShield's methodology and provides an open foundation for further research.

## Future Research Directions

Identified opportunities for extension:
- Systematic comparison across model sizes and architectures
- Ablation studies quantifying individual component contributions
- Extended evaluation on diverse datasets and real-world attacks
- Adversarial training for improved robustness
- Model distillation for reduced inference latency
- Multilingual detection capabilities
- Ensemble approaches for enhanced performance

## Technical Artifacts

The repository includes:
- Complete training pipeline (Jupyter notebook)
- Trained model weights and configuration
- Evaluation results at all FPR thresholds
- Visualization of ROC curves emphasizing low FPR region
- Temperature scaling implementation
- Comprehensive documentation

## Citation

Jacob, D., Alzahrani, H., Hu, Z., Alomair, B., & Wagner, D. (2025). PromptShield: Deployable Detection for Prompt Injection Attacks. ACM Conference on Data and Application Security and Privacy (CODASPY).
