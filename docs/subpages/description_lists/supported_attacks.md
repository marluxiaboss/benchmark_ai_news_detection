# List of supported evasion attacks

See `detector_benchmark/generation/attack_loader.py` and `detector_benchmark/conf/generation`

```{admonition} Supported evasion attacks
- "no_attack": Normal generation.
- "prompt_attack": Evasion attack where the attacker provide a specific prompt used to fool the detector.
- "gen_params_attack": Evasion attack where the attacker uses specific generation parameters (temperature for example) to fool the detector.
- "prompt_paraphrasing_attack": Evasion attack where the output of the LLM is paraphrased with another LLM using a paraphrasing prompt (same LLM used for generation only for now).
```