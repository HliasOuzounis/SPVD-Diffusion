# Missing

## Questions

Variance in results? Try set seed
Lower steps sometimes better mean CD and EMD for skip steps
Use half of tests for chair

## Training

<!-- - Unconditional training
        - Distillation
            - Car (500, 250, 125, 63, 32, 16, 8, 4, 2, 1 steps) Missing
            - Airplane (500, 250, 125, 63, 32, 16, 8, 4, 2, 1 steps) Missing
            - Chair (500, 250, 125, 63, 32, 16, 8, 4, 2, 1 steps) Missing
        - Retrained
            - Car (500, 250, 125, steps) Missing
            - Airplane (500, 250, 125, steps) Missing
            - Chair (500, 250, 125, steps) Missing -->

- Conditional training
        - Distillation
            - Car **Complete** (run all again cause skip is better) (og was on 5000 epochs)
            - Airplane (8, 4, 2, 1 steps) Missing (og was on 3750 epochs)
            - Chair (8, 4, 2, 1 steps) Missing (og was on 2500 epochs)
        - Retrained
            - Car (500, 250, 125) **Complete**
            - Airplane (500, 250) **Complete**
            - Chair **Complete**

## Evaluation

✈️ Airplane
    - DDIM
        - 1000 steps
            - Conditional: Missing
            - Unconditional: Missing
        - Distillation (500 → 1 steps)
            - Conditional (500, 16, 8, 4, 2, 1): Missing
            - Unconditional: Missing
        - Skip Steps (500 → 1 steps)
            - Conditional: ✅ Complete
            - Unconditional: ✅ Complete
    - DDPM
        - 1000 steps
            - Conditional: ✅ Complete
            - Unconditional: Missing
        - Distillation (500 → 1 steps)
            - Conditional: Missing
            - Unconditional: Missing
        - Retrained (500, 250 steps)
            - Conditional: Missing
            - Unconditional: Missing

🚗 Car
    - DDIM
        - 1000 steps
            - Conditional: ✅ Complete
            - Unconditional: Missing
        - Distillation (500 → 1 steps)
            - Conditional (2, 1): Missing
            - Unconditional: Missing
        - Skip Steps (500 → 1 steps)
            - Conditional: ✅ Complete
            - Unconditional: ✅ Complete
    - DDPM
        - 1000 steps
            - Conditional: ✅ Complete
            - Unconditional: Missing
        - Distillation (500 → 1 steps)
            - Conditional: ✅ Complete
            - Unconditional: Missing
        - Retrained (500, 250, 125 steps)
            - Conditional: ✅ Complete
            - Unconditional: Missing

🪑 Chair
    - DDIM
        - 1000 steps
            - Conditional: Missing
            - Unconditional: Missing
        - Distillation (500 → 1 steps)
            - Conditional (500, 16, 8, 4, 2, 1): Missing
            - Unconditional: Missing
        - Skip Steps (500 → 1 steps)
            - Conditional: ✅ Complete
            - Unconditional: Missing
    - DDPM
        - 1000 steps
            - Conditional: Missing
            - Unconditional: Missing
        - Distillation (500 → 1 steps)
            - Conditional: Missing
            - Unconditional: Missing
        - Retrained (500, 250, 125 steps)
            - Conditional: Missing
            - Unconditional: Missing

Random Noise (Missing)

## Running

Distilling airplane with 8, 4, 2, 1 steps
Running with distilled=False, scheduler=ddim, conditional=False, on_all=True for steps (500, 250, 125, 63, 32, 16, 8, 4, 2, 1) Chair. (chair)
Distilling airplane with 250 steps, 3750 epochs (2 Days finished: Sunday) (Base)

## Extra

Check translations on dataset

- Mean
- Std
- Std of mean (calculate mean of every model, then calculate std of means)

Maybe on chair evaluate on half the test set (704)
Maybe evaluation with a set seed for each method for fair comparison

## Time

Evaluation time

- Chair (10 x 3 x 2) = 60 hours
- Airplane (10 x 0.5 x 2 x 3) = 30 hours
- Car (10 x 0.5 x 2 x 3) = 30 hours

Training

- 1000 epochs = ~8 hours
- 3750 epochs = ~25 hours

Training Retrained

- 1000 epochs = 6 hours
