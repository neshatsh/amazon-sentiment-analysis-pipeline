## Final Results Summary

| Activation Function | Best L2 Regularization | Best Dropout Rate | Validation Accuracy | Test Accuracy |
|---------------------|------------------------|-------------------|---------------------|---------------|
| ReLU                | 0.001                  | 0.5               | 0.7507              | 0.7503        |
| Sigmoid             | 0.001                  | 0.3               | 0.735               | 0.7331        |
| Tanh                | 0.001                  | 0.3               | 0.7444              | 0.7475        |

## Analysis 

### Effect of Activation Functions

Among the three activation functions, ReLU consistently performed best, achieving the highest validation (0.7507) and test accuracy (0.7503). Tanh came close but slightly underperformed compared to ReLU. Sigmoid lagged behind, likely due to slower convergence and saturation issues.

### Effect of L2 Regularization

L2 regularization helped improve generalization by preventing the model from overfitting on the training data. The experiments showed that a lower L2 value (0.001) consistently outperformed a higher one (0.01), suggesting that too much regularization degraded model capacity. The best performing models across all activation functions used L2=0001.

### Effect of Dropout

Dropout added beneficial noise during training, helping the models avoid overfitting. The optimal dropout rate varied slightly by activation, but 0.3–0.5 generally worked best. For ReLU, a higher dropout rate (0.5) achieved the highest performance, suggesting robustness to regularization.

### Best Configuration and Key Insight

Best Model: ReLU with L2=0.001, Dropout=0.5, Test Accuracy=0.7503
This configuration worked best likely due to ReLU's strong gradient propagation and the high dropout rate mitigating overfitting. The most important insight was how small changes in regularization and activation choice significantly impacted generalization performance, even in relatively simple feedforward networks.