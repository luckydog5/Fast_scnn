# Fast_scnn

Initial learning rate 0.045, use SGD optimizer and "polynomialdecay" schedule.

After 100 epochs, reduce learning rate to 0.00045, otherwise loss will rise up.

When classes number = max(image) 150+1, loss will be 1.78 with acc = 0.62, val_acc = 0.42.

Meanwhile, it is hard to reduce the loss.


When classes number = 21, thing seams good and ideal. But i think it still has class imbalance 

issue.




400 epochs acc:0.68 val_acc:0.51
