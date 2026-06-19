# Parameter to test
- Epoch (max_epoch =  500; patience = 30 - when no improvement, then pick the winner)

- Attention Heads (K --> 4;12)
- Hidden Feature Dimensions (F' --> 1; 3 )

- Embedding Vector Size (--> 4;12)
- Attention Vector Size (--> 4;12)

- Training intensity (t per epoch; memory used; flops (sum of additions + multiplications); parameter total)
- Accuracy (mean)
- Overfitting (When test and trainings accuracy worsen; generalization_gap = loss_test - loss_training)
- Oversmoothing (How similar the nodes became to each other - Mean Absolute Distance?)


# Next Steps 
1. Model stabil aufsetzen
2. KPIs definieren
3. Parameter in Skripten einstellen 
4. Experimente durchlaufen lassen 
5. Presentation vorstellen