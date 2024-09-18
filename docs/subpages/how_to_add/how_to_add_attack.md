# How to add an attack
To add an attack, 3 files need to be added/modified:
- add: a .py file in `generation` containing the class for the attack, inheriting the base ArticleGenerator class
- add: a configuration file under `detector_benchmark/conf/generation` to configure the new attack.
- modify: `detector_benchmark/generation/attack_loader.py` to be able to load the attack.

The added detector class should have at least a constructor and a `generate_adversarial_text` function with the following signature:¨

``` python
def generate_adversarial_text(self, prefixes: list[str], batch_size: int) -> list[str]:
```

Where `prefixes` is a list of input texts to continue for the generation (see how the dataset samples look like) and it should return the list of generated text which is the continuation of the prefixes.

You can look at already added detectors for guiding examples.

Note: for any attack that involves either using a specific prompt for the generation or modifying a prompt parameter, the already existing attacks in `detector_benchmark/generation/gen_params_attack.py` and `detector_benchmark/generation/prompt_attack.py` can be used for this purpose by only modifying the related configuration file in `conf`.