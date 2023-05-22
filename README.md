# GraphRXN
Source code for our paper "A deep learning framework for accurate reaction prediction and its application on high-throughput experimentation data".
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

### **Model performance of in-house dataset over 5-fold CV on test set**
Groupe | Size | methods | R2 | MAE | RMSE
:---: |:----:| :---: | :---: | :---: | :---:
Entire | 1558 | GraphRXN-concat | 0.713 | 0.06 | 0.09
Entire | 1558 | GraphRXN-sum | 0.704 | 0.06 | 0.09
Entire | 1558 | Yield-BERT | 0.645 | 0.10 | 0.07
Entire | 1558 | DeepReac+ | 0.610 | 0.07 | 0.10
G1 | 317 | GraphRXN-concat | 0.661 | 0.08 | 0.11
G1 | 317 | GraphRXN-sum | 0.462 | 0.11 | 0.14
G1 | 317 | Yield-BERT | 0.718 | 0.07 | 0.10
G1 | 317 | DeepReac+ | 0.551 | 0.09 | 0.13
G2 | 419 | GraphRXN-concat | 0.629 | 0.05 | 0.07
G2 | 419 | GraphRXN-sum | 0.592 | 0.06 | 0.07
G2 | 419 | Yield-BERT | 0.512 | 0.06 | 0.08
G2 | 419 | DeepReac+ | 0.528 | 0.06 | 0.08
G3 | 401 | GraphRXN-concat | 0.802 | 0.06 | 0.08
G3 | 401 | GraphRXN-sum | 0.775 | 0.06 | 0.08
G3 | 401 | Yield-BERT | 0.785 | 0.06 | 0.08
G3 | 401 | DeepReac+ | 0.745 | 0.07 | 0.09
G4 | 421 | GraphRXN-concat | 0.459 | 0.08 | 0.12
G4 | 421 | GraphRXN-sum | 0.419 | 0.09 | 0.12
G4 | 421 | Yield-BERT | 0.503 | 0.08 | 0.11
G4 | 421 | DeepReac+ | 0.23 | 0.10 | 0.14

## Quick start

### GraphRXN 
    conda env create -f GraphRXN.yaml ### Create GraphRXN env
    conda activate GraphRXN
    python reaction_train.py  --data_path data_scaler/Buchward-Hartwig/random_split/FullCV_01_train_temp_scaler.csv
                              --separate_test_path data_scaler/Buchward-Hartwig/random_split/FullCV_01_test_temp_scaler.csv
                              --dataset_type regression 
                              --num_folds 1 
                              --gpu 0 
                              --epochs 100 
                              --batch_size 128 
                              --save_dir ./result/Buchward/concat_01_temp
                              --metric r2 
                              --reaction_agg_method concat
    Note: If choosing summation aggregation method, please specify --reaction_agg_method sum

### DeepReac+ 
    cd DeepReac
    conda env create -f DeepReact.yaml ### Create DeepReac+ env
    conda activate DeepReact
    ### train and predict
    python DeepReac_train.py -train data_scaler/Buchward-Hartwig/random_split/FullCV_01_train_temp_scaler.csv
                             -test data_scaler/Buchward-Hartwig/random_split/FullCV_01_test_temp_scaler.csv
                             -epochs 100
                             -stats ./result_scaler/Buchward_01_test_stats.csv

### Yield-BERT
    cd Yield-BERT
    conda env create -f rxnyields.yaml ### create Yield-BERT env
    conda activate rxnyields

    cd yield-BERT_baseline

    ### For Dataset 1 (Buchwald) training
    python launch_buchwald_hartwig_training.py
    
    ### For Dataset 2 (Suzuki) training
    python lauch_suzuki_miyaura_training.py
    
    ### For Dataset 3 (Denmark) training
    python data3_training_10cv.py
    
    ### For in-house dataset training
    python inhouse_data_transform.py
    
    
    
    





