# GraphRXN
Source code for our paper "	GraphRXN: A Novel Representation for Reaction Prediction".
The code was built based on CMPNN (https://github.com/SY575/CMPNN), DeepReac (https://github.com/bm2-lab/DeepReac), YieldBert (https://github.com/bm2-lab/DeepReac).
Thanks a lot for their sharing.


### **Figure 1. Model architecture of GraphRXN</center>**

![Figure1](https://github.com/jidushanbojue/GraphRXN/blob/master/picture/Figure1.png "Figure1")

### **Figure 2. General workflow of HTE process**
![Figure2](https://github.com/jidushanbojue/GraphRXN/blob/master/picture/Figure2.png "Figure2")

### **Figure 3. Reaction scheme and substrate scope**
![Figure3](https://github.com/jidushanbojue/GraphRXN/blob/master/picture/Figure3.png "Figure3")

### **Figure 4. Distribution of Ratio(UV), where A represents amine, and B represents bromide**
![Figure4](https://github.com/jidushanbojue/GraphRXN/blob/master/picture/Figure4.png "Figure4")

### **Figure 5. The scatter plots of GraphRXN on the entire dataset**
![Figure5](https://github.com/jidushanbojue/GraphRXN/blob/master/picture/Figure5.png "Figure5")

### **Model Performance of three public datasets over ten-fold CV on test set**

Dataset | Methods | R2 | MAE | RMSE
:---: | :---: | :---: | :---: | :---:
Dataset1 | GraphRXN-concat | 0.951 | 4.30 | 5.98
Dataset1 | GraphRXN-sum | 0.937 | 4.85 | 6.80
Dataset1 | Yield-BERT | 0.951 | 4.00 | 6.03
Dataset1  | DeepReac+ | 0.922 | 5.25 | 7.54
Dataset2 | GraphRXN-concat | 0.844 | 7.94 | 11.08
Dataset2 | GraphRXN-sum | 0.838 | 8.09 | 11.29
Dataset2 | Yield-BERT | 0.815 | 8.13 | 12.08
Dataset2 | DeepReac+  | 0.827 | 8.06 | 11.65
Dataset3 | GraphRXN-concat | 0.892 | 0.16 | 0.23
Dataset3 | GraphRXN-sum | 0.881 | 0.18 | 0.24
Dataset3 | Yield-BERT | 0.886 | 0.16 | 0.24
Dataset3 | DeepReac+ | 0.853 | 0.18 | 0.25




