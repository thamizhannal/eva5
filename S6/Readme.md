### Objective:

Have reused session5 best model and applied following changes into that.

1. with L1 + BN

2. with L2 + BN

3. with L1 and L2 with BN

4. with GBN

5. with L1 and L2 with GBN

   

### Target:

Have reached 99.58% validation accuracy for L2+BN, with GBN & L1&L2 with GBN . 



### Results:

1. Parameters: 8002
2. Best Train Accuracy: 99.08%
3. Best Test Accuracy: 99.58%
4. greater than 99.4% in consistently >4 epochs

### Analysis:

**Lambda used for L1 loss is 0.001**. Have used grid search and experimented L1 loss with various values such as 0.01, 0.03, 0.001, 0.003, 0.0001 & 0.0003 and found that 0.001 was optimal and have got validation accuracy as 99%.  

**Weigh decay used for L2 is 0.0001.** We used grid search and experimented weight decay various weight decay values 0.01, 0.03, 0.001, 0.003, 0.0001 & 0.0003 and seen that optimal value was  0.0001.

### Loss Curve Analysis:

 **L1 loss curve:** It is fluctuating a lot, not smoother and reduced slightly from where it has started. Loss started with 0.2 and increased to 0.3 at 8th epoch, after that stated reducing gradually and reached to 0.08.

 **L1 & L2 Loss curve:** This loss also looks much similar to L1 loss function but fluctuating slightly lesser as compare to L1 loss. 

 **L2 Loss, GBN Loss & L1L2GBN loss curve** looks much similar and smoother. Loss starts reducing as epoch progresses and after a point loss between these models looks narrow and like single line.


#### Accuracy Curve 

**L1 accuracy and L1&L2 accuracy curves ** looks similar, but L1 accuracy curve fluctuates a lot compare to L1L2 curve.

L1  and L1&L2 regularization both methods does not improves accuracy much. After 25 epochs it improved it improved accuracy by 2%. (96% at 1st epoch and 98% at 25th epoch)

Seems L1 Regularization create more fluctuation in loss & accuracy curve and does not improve model much.

**L2 Regularization**: This curve is very smooth, gradually increasing and  more consistent. It has started at 98% at 1st epoch and start improving gradually and reached 99.58% at the end of 25th epoch. 

**L2 Reg, GBN Only and L1L2 with GBN validation accuracies**  curve looks much smoother, consistently improving. 

**Best Model:**  **Thought L2 Reg, GBN Only and L1L2 with GBN validation accuracies exhibits similar validation accuracies,  L1L2 with GBN outperform all other models  it reached validation highest validation accuracy 99.5% at train accuracy of 99.1%. Looks this model has more scope to push validation accuracy further**. 