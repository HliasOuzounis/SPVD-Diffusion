# TODO

Μιχαλάκσης
Παπούλιας


## Questions

- Pre trained good unconditional?
- DDIM skip steps is almost same as distillation. What do?
  - Ideas why it could be better
    - Can use distilled network with DDPM. (Only time embedding matters)
    - Distilled network learns to generate more details. Details are only added in the later steps, so skipping them might lead to worse results. On the other hand, distillation can capture the details and move them to an earlier step.
  - Pray that metrics are better for distillation.
  - Train on waaaay less epochs (100, 200) and see.
- What if results from other methods are better?

## Training

Train on higher unconditional %

Train a new model with 500 steps of noising/denoising from scratch on same epochs. Comparable
Train a new model with 250 steps of noising/denoising from scratch on same epochs.
Train a new model with 125 steps of noising/denoising from scratch on same epochs.

Try 500 epochs

## Evaluation

Evaluate unconditional model. Compare 1-NN with SOTA.

Evaluate conditional model. Compare 1-NN with SOTA. Non normalized will be worse because need to match the translation.
Normalized will be better, no translation needed + already given distribution in image space, need to translate to 3d space.
More practical applications.

Conditional: Compare mean CD and mean EMD with SOTA. (normalized or standardized)
Compare F1 score (τ=0.01) with SOTA. (normalized or standardized)

## Results (ideally)

Distillation better than training from scratch and DDIM skip steps. (At least for lower step count)
Conditional generation good.

