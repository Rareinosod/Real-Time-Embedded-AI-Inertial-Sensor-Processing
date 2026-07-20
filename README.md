# Real-Time-Embedded-AI-Inertial-Sensor-Processing

Pipeline in MATLAB/Simulink per l'acquisizione real-time ad altissima frequenza di dati inerziali da dispositivi mobili Android. 
* Progettazione del protocollo di raccolta (dataset su oltre 200 soggetti), data cleaning e feature engineering per pattern motori. 
* Addestramento comparativo SVM vs 1D CNN, con implementazione finale della CNN 1D.

---

## Struttura del Repository e Descrizione dei Moduli

Il progetto è architettato in tre sezioni principali che separano l'acquisizione grezza, l'elaborazione e addestramento offline, e infine il deployment per l'inferenza online in tempo reale.

### 1. Acquisizione Dati (Root / Primi File)
Questa sezione contiene i file necessari per la cattura e l'estrazione dei segnali inerziali direttamente dai dispositivi hardware.

* **`simulink/systems/system_root.xml` (e configurazioni collegate):** Modello Simulink configurato per il deployment su hardware Android (`Android Device`). Utilizza i blocchi `androidsensorlib` per acquisire i dati di Accelerometro e Giroscopio (3 assi ciascuno) con un tempo di campionamento fissato a 1/200 s (200 Hz). I segnali vengono uniti tramite un Mux e inviati al Workspace di MATLAB in una struttura temporale (`datiSensori`).
* **Script di Estrazione Dati (`Estrazione_dati.m`):** Script MATLAB che accede alla struttura `datiSensori` generata da Simulink, separa i 6 canali (Acc X,Y,Z e Gyr X,Y,Z) e genera un plot grafico rapido per la verifica visiva. Successivamente, si occupa di salvare automaticamente le acquisizioni in formato `.mat` all'interno di un'apposita sottocartella denominata "Prove", richiedendo all'utente il nome specifico per ogni test (es. `Soggetto1_attività1`).

---

### 2. Cartella `OFFLINE` (Preprocessing e Modelli)
Questa cartella contiene l'intera pipeline di data cleaning, feature engineering e il training comparativo dei modelli di Machine e Deep Learning.

* **Script di Preprocessing e Pulizia (`Preprocessing.m`):** Elabora i dati grezzi applicando un filtro passa-banda Butterworth (0.5 Hz - 20 Hz) e un filtro di smoothing (`ALPHA_SMOOTH = 0.2`) per abbattere il rumore. Implementa un algoritmo di trigger personalizzato (basato su Media Mobile Esponenziale - EMA) sull'asse Y dell'accelerometro per isolare e ritagliare le finestre temporali esatte dei colpi (-1 secondo prima e +1 secondo dopo l'impatto). Salva le finestre normalizzate (Z-score) pronte per le reti neurali.
* **Script Estrazione Features per SVM (`Estrazione_features.m`):** Processa i segnali puliti ed estrae matematicamente 10 feature statistiche (Media, Deviazione Standard, Varianza, Skewness, Kurtosis, SumSquares, RMS, Massimo, Minimo, Range) per ciascuno dei 6 assi sensoriali, generando un dataset tabulare denso di 60 colonne.
* **Script PCA e Addestramento SVM (`PCA_SVM.m`):** Carica il dataset delle features estratte (suddiviso in 70% Train, 30% Test), applica un'ulteriore standardizzazione e calcola la Principal Component Analysis (PCA) mantenendo il 95% della varianza spiegata. Addestra un classificatore Support Vector Machine (SVM) con kernel lineare per discriminare i colpi (Jab vs Montante) e salva il modello ottimizzato in `modello_PCA_SVM_VALIDO.mat` per l'uso embedded.
* **Script Addestramento 1D CNN (`CNN_1D.m`):** Costruisce e addestra una Rete Neurale Convoluzionale monodimensionale operante direttamente sulle serie temporali grezze (6 canali). L'architettura prevede livelli di convoluzione (`convolution1dLayer`), Batch Normalization, ReLU, Max Pooling e Global Average Pooling, terminando con un classificatore Softmax a 2 classi. Calcola la matrice di confusione, precision, recall e F1-score, salvando la rete finale per Simulink (`Rete_CNN_Finale.mat`).

---

### 3. Cartella `ONLINE` (Inferenza Real-Time)
Questa cartella contiene gli applicativi per la fase di test e inferenza dal vivo.

* **Modello Simulink per Inferenza (CNN Pipeline):** Integra il modello 1D CNN esportato dalla fase Offline (`Rete_CNN_Finale.mat` o equivalente) all'interno dell'ambiente Simulink. Si occupa di ricevere il flusso di dati inerziali in real-time dallo smartphone e processarli istantaneamente per restituire la classificazione del pattern motorio (Jab o Montante) a runtime, dimostrando l'efficacia del deployment su architetture Embedded.
* **Script e Dati per i Test Online:** Set di script accessori volti a validare la robustezza e la latenza del modello direttamente sul campo (Online Testing).
