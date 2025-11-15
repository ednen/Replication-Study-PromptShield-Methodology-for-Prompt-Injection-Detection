# Attribution and Usage Guidelines

## License

This project is released under the MIT License. See the LICENSE file for full details.

## What the MIT License Means

The MIT License is a permissive open-source license that allows others to:
- Use this code for any purpose (commercial or non-commercial)
- Modify and adapt the code
- Distribute the code and modifications
- Include the code in proprietary software

However, they MUST:
- Include the original copyright notice and license text in any copies or substantial portions
- Not hold you liable for any issues with the code

## Proper Attribution

If you use this work in your research, development, or production systems, please provide proper attribution:

### For Academic Papers

```bibtex
@misc{promptshield_replication_2025,
  title={PromptShield Replication: Prompt Injection Detection with Llama 3.1},
  author={Ozan Bülen},
  year={2025},
  howpublished={\url{https://github.com/ednen/Replication-Study-PromptShield-Methodology-for-Prompt-Injection-Detection},
  note={Implementation replicating PromptShield methodology for prompt injection detection}
}
```

### For Software Projects

Include in your README or documentation:

```
This project uses code from PromptShield Replication by Ozan Bülen
https://github.com/ednen/Replication-Study-PromptShield-Methodology-for-Prompt-Injection-Detection
Licensed under MIT License
```

### For Commercial Use

While the MIT License permits commercial use, we appreciate attribution:
- Link to this repository in your documentation
- Mention the use of this implementation in your security disclosures
- Consider contributing improvements back to the community

## What Constitutes Proper Use

### Acceptable Use

1. Using the code to implement prompt injection detection in your applications
2. Modifying the code to adapt it for your specific use case
3. Building upon this work for further research
4. Including the code in commercial products (with proper attribution)
5. Using the methodology and results for academic research

### Required Practices

1. Include the MIT License text with any distribution
2. Preserve copyright notices in source files
3. Cite this work in academic publications that build upon it
4. Acknowledge the original PromptShield work by Wagner et al.

### Discouraged Practices (though not legally prohibited)

1. Claiming this work as your own without attribution
2. Removing copyright notices or license information
3. Using the code without understanding its limitations
4. Deploying in production without proper testing and validation

## Contributing Back

While not required by the license, we encourage contributions:
- Report bugs or issues
- Submit improvements via pull requests
- Share your results if you extend this work
- Help improve documentation

## Model Weights and Data

Note that while this code is MIT licensed:
- Llama 3.1 model weights are subject to Meta's license agreement
- The PromptShield dataset is subject to its own license terms
- You must comply with all applicable licenses when using this software

## Disclaimer

This software is provided "as is" without warranty. See the LICENSE file for full disclaimer text. Specifically:
- No guarantee of detection accuracy in production environments
- No warranty against adversarial attacks or evasion
- Users are responsible for validating performance in their specific context
- The authors are not liable for security breaches or false negatives

## Questions About Usage

For questions about proper attribution or usage:
- Open an issue on GitHub
- Contact ozanbulen@gmail.com
- Review the LICENSE file for legal details

## Acknowledgment of Sources

This work builds upon:
- PromptShield methodology (Jacob et al., 2025)
- Llama model family (Meta AI)
- LoRA technique (Hu et al., 2021)
- Temperature scaling approach (Guo et al., 2017)

All original sources should be properly cited when using this work.
Jacob, D., Alzahrani, H., Hu, Z., Alomair, B., & Wagner, D. (2025). PromptShield: Deployable Detection for Prompt Injection Attacks. ACM Conference on Data and Application Security and Privacy (CODASPY).
