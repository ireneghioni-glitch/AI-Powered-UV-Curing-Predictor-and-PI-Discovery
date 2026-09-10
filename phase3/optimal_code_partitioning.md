## 🧩 Cosa e Come si potrebbe suddividere
| Modulo | Contenuto | Perché separarlo |
| --- | --- | --- |
| data_loader.py | Caricamento embeddings, creazione dataset combinato, DataLoader | Se devi caricare dati in più script (es. per inference)
| train_utils.py | Funzioni train_epoch, evaluate, evaluate_metrics | Riutilizzabili in altri script di addestramento
| config.py | Parametri (BATCH_SIZE, NUM_EPOCHS, percorsi) | Centralizzare le configurazioni per modificarle facilmente
| visualize.py | Funzione per creare i grafici | Se vuoi fare più plot in diversi contesti