# Metrics Definition (v1)

## Precision

For class *c*, precision is the fraction of predictions labeled *c* that are correct.

## Recall

For class *c*, recall is the fraction of true *c* samples that are recovered by predictions.

## F1 Score

For class *c*, F1 is the harmonic mean of precision and recall.

## Confusion Matrix

A square matrix where rows represent true labels and columns represent predicted labels for the v1 class set.

## Uncertain Rate

The proportion of evaluated samples predicted as `uncertain`.

## Selective Accuracy

Accuracy computed only over non-`uncertain` predictions, useful for evaluating abstention behavior.
