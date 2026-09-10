## Opzione C – Tuning a "piramide" (strategia professionale)
Primo livello (griglia larga): max_depth, learning_rate, reg_lambda.

Secondo livello (affina): usa i migliori valori del primo livello e aggiungi subsample e colsample_bytree.

Terzo livello (fine tuning): fissa i parametri e ottimizza reg_alpha.

Esempio:

```python
# Livello 1
param_grid_1 = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.2],
    'reg_lambda': [0.5, 1.0, 2.0]
}
# ... trova best_params_1 ...

# Livello 2 (usa i migliori di livello 1)
param_grid_2 = {
    'max_depth': [best_params_1['max_depth']],
    'learning_rate': [best_params_1['learning_rate']],
    'reg_lambda': [best_params_1['reg_lambda']],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}
# ... trova best_params_2 ...

# Livello 3 (fine tuning)
param_grid_3 = {
    'max_depth': [best_params_2['max_depth']],
    'learning_rate': [best_params_2['learning_rate']],
    'reg_lambda': [best_params_2['reg_lambda']],
    'subsample': [best_params_2['subsample']],
    'colsample_bytree': [best_params_2['colsample_bytree']],
    'reg_alpha': [0, 0.05, 0.1, 0.2]
}
```