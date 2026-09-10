# Outputs log of XGBoost model
## 1st version
Whith TRAIN section:
```python
# ==================== TRAIN XGBOOST ====================
'''
XGBoost has many hyperparameters. 
We'll start with a baseline configuration and then tune.
'''
print("Training XGBoost regressor with early stopping...")

model = xgb.XGBRegressor(
    n_estimators=1000,                          # number of trees, high because we have Early Stopping
    learning_rate=0.1,                          # step size shrinkage
    max_depth=6,                                # maximum tree depth
    subsample=0.8,                              # fraction of samples used per tree
    colsample_bytree=0.8,                       # fraction of features used per tree
    reg_alpha=0.1,                              # L1 regularization
    reg_lambda=1.0,                             # L2 regularization
    random_state=SEED if SEED > 0 else None,
    verbosity=0                                 # suppress training messages
)

# introduce Early Stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],          # Validation set for monitoring
    early_stopping_rounds=10,           # Stops if loss doesn't get better after 10 rounds
    verbose=True
)

print(f"Training complete. Best number of trees: {model.best_iteration}")
```
Gave this output:
```text
(chemvision) PS D:\Irene\Desktop\AI_&_Data_Science_training_BeCode\BeCode_Projects\specialization\AI-Powered-UV-Curing-Predictor-and-PI-Discovery> python phase4/train_xgboost.py
[INFO] Fixed seed: 42 (reproducible results)
Loading of PIs embeddings...
Loading of Monomers embeddings...
PIs and Monomers embeddings loaded successfully.
    PIs embeddings: (208, 1280)
    Monomers embeddings: (40, 1280)
Creating the combined dataset...
Visual embeddings shape: (8320, 2560)
Generating environmental features...
Environmental features shape: (8320, 4)
Final feature matrix shape: (8320, 2564)
Generating simulated target values...
Target values shape: (8320,)
Min: 1.69%, Max: 93.00%
Train set: 5824 samples
Validation set: 1248 samples
Test set: 1248 samples
Training XGBoost regressor with early stopping...
[0]     validation_0-rmse:17.20530
[1]     validation_0-rmse:16.05829
[2]     validation_0-rmse:15.22510
[3]     validation_0-rmse:14.54637
[4]     validation_0-rmse:13.58890
[5]     validation_0-rmse:12.94618
[6]     validation_0-rmse:12.38446
[7]     validation_0-rmse:11.66756
[8]     validation_0-rmse:11.23338
[9]     validation_0-rmse:10.85804
[10]    validation_0-rmse:10.28930
[11]    validation_0-rmse:9.98098
[12]    validation_0-rmse:9.51832
[13]    validation_0-rmse:9.07447
[14]    validation_0-rmse:8.85360
[15]    validation_0-rmse:8.48220
[16]    validation_0-rmse:8.15767
[17]    validation_0-rmse:7.86548
[18]    validation_0-rmse:7.71701
[19]    validation_0-rmse:7.47219
[20]    validation_0-rmse:7.27133
[21]    validation_0-rmse:7.07509
[22]    validation_0-rmse:6.91868
[23]    validation_0-rmse:6.82191
[24]    validation_0-rmse:6.67977
[25]    validation_0-rmse:6.57500
[26]    validation_0-rmse:6.45751
[27]    validation_0-rmse:6.39519
[28]    validation_0-rmse:6.29898
[29]    validation_0-rmse:6.24782
[30]    validation_0-rmse:6.17418
[31]    validation_0-rmse:6.10296
[32]    validation_0-rmse:6.06646
[33]    validation_0-rmse:6.02558
[34]    validation_0-rmse:5.97620
[35]    validation_0-rmse:5.93311
[36]    validation_0-rmse:5.89264
[37]    validation_0-rmse:5.85978
[38]    validation_0-rmse:5.82995
[39]    validation_0-rmse:5.81274
[40]    validation_0-rmse:5.77288
[41]    validation_0-rmse:5.76254
[42]    validation_0-rmse:5.74405
[43]    validation_0-rmse:5.71394
[44]    validation_0-rmse:5.69461
[45]    validation_0-rmse:5.67929
[46]    validation_0-rmse:5.65494
[47]    validation_0-rmse:5.64252
[48]    validation_0-rmse:5.63678
[49]    validation_0-rmse:5.63856
[50]    validation_0-rmse:5.63082
[51]    validation_0-rmse:5.62347
[52]    validation_0-rmse:5.61480
[53]    validation_0-rmse:5.60660
[54]    validation_0-rmse:5.60184
[55]    validation_0-rmse:5.59987
[56]    validation_0-rmse:5.59290
[57]    validation_0-rmse:5.58403
[58]    validation_0-rmse:5.57381
[59]    validation_0-rmse:5.56370
[60]    validation_0-rmse:5.56405
[61]    validation_0-rmse:5.56214
[62]    validation_0-rmse:5.55814
[63]    validation_0-rmse:5.53953
[64]    validation_0-rmse:5.52705
[65]    validation_0-rmse:5.52384
[66]    validation_0-rmse:5.52187
[67]    validation_0-rmse:5.52414
[68]    validation_0-rmse:5.52196
[69]    validation_0-rmse:5.51899
[70]    validation_0-rmse:5.51845
[71]    validation_0-rmse:5.51706
[72]    validation_0-rmse:5.50446
[73]    validation_0-rmse:5.50141
[74]    validation_0-rmse:5.49964
[75]    validation_0-rmse:5.50543
[76]    validation_0-rmse:5.50667
[77]    validation_0-rmse:5.50702
[78]    validation_0-rmse:5.50305
[79]    validation_0-rmse:5.50457
[80]    validation_0-rmse:5.50255
[81]    validation_0-rmse:5.50446
[82]    validation_0-rmse:5.49794
[83]    validation_0-rmse:5.48799
[84]    validation_0-rmse:5.48652
[85]    validation_0-rmse:5.48840
[86]    validation_0-rmse:5.48986
[87]    validation_0-rmse:5.49050
[88]    validation_0-rmse:5.49254
[89]    validation_0-rmse:5.49098
[90]    validation_0-rmse:5.49005
[91]    validation_0-rmse:5.48685
[92]    validation_0-rmse:5.48918
[93]    validation_0-rmse:5.49105
[94]    validation_0-rmse:5.49102
Training complete. Best number of trees: 84
Evaluating the XGBoost model...
=== Test Set Metrics ===
MSE: 29.6092
RMSE: 5.4414
R²: 0.9051
MAE: 4.3086
=== Comparison with Phase 3 Neural Network ===
Neural Network (Phase 3):
  MSE: 180.0892
  RMSE: 13.4197
  R²: 0.4228
  MAE: 11.1790
XGBoost (Phase 4):
  MSE: 29.6092
  RMSE: 5.4414
  R²: 0.9051
  MAE: 4.3086
XGBoost improves MSE by: 83.56%
Plot saved to D:\Irene\Desktop\AI_&_Data_Science_training_BeCode\BeCode_Projects\specialization\AI-Powered-UV-Curing-Predictor-and-PI-Discovery\phase4\visuals\comparison_plots.png
XGBoost model saved to D:\Irene\Desktop\AI_&_Data_Science_training_BeCode\BeCode_Projects\specialization\AI-Powered-UV-Curing-Predictor-and-PI-Discovery\phase4\xgboost_model.json
Model config saved to model_config.json
```
It's a good result.

Now I implement hyperparameter tuning with **GridSearch** and I have to make it compatible with the Early Stopping strategy.  
A wrapper class should:  
1. receives the `model`, `eval_set` and `early_stopping_rounds`.
2. does override of fit() method to include early stopping.
3. gives back the trained model with optimal number of trees.

