clc; clear; close all;

%% 1. CONFIGURAZIONE PERCORSO
BASE_DATA_FOLDER = '/Users/alessandro/Desktop/AUTOMAZIONE 2/OFFLINE/DATASET_PROCESSATO_FINALE';

if ~isfolder(BASE_DATA_FOLDER)
    error('ERRORE CRITICO: Cartella non trovata in: %s', BASE_DATA_FOLDER);
end

CLASSES = {'Jab', 'Montante'};
LABELS  = [0, 1]; 

fprintf('--- AVVIO CARICAMENTO DATI ---\n');
XCell = {}; 
y_vec = []; 
total_files = 0;

% --- LOOP CARICAMENTO ---
for c = 1:length(CLASSES)
    className = CLASSES{c};
    classLabel = LABELS(c);
    
    classDir = fullfile(BASE_DATA_FOLDER, className);
    subjects = dir(classDir);
    subjects = subjects([subjects.isdir]); 
    subjects = subjects(~ismember({subjects.name}, {'.', '..'})); 
    
    for s = 1:length(subjects)
        subjName = subjects(s).name;
        datiDir = fullfile(classDir, subjName, 'dati');
        
        if ~isfolder(datiDir), continue; end
        
        matFiles = dir(fullfile(datiDir, '*.mat'));
        
        for k = 1:length(matFiles)
            filePath = fullfile(datiDir, matFiles(k).name);
            tmp = load(filePath); 
            
            if isfield(tmp, 'finestra')
                dat = tmp.finestra;
                % Unione Acc+Gyr e Trasposizione [6 x Tempo]
                full_signal = [dat.acc_norm, dat.gyr_norm]'; 
                
                XCell{end+1} = full_signal; %#ok<SAGROW>
                y_vec(end+1) = classLabel;  %#ok<SAGROW>
                total_files = total_files + 1;
            end
        end
    end
end

if total_files == 0
    error('Nessun file trovato! Controlla il percorso.');
end

fprintf('Caricamento completato: %d campioni.\n', total_files);
% Conversione in Categorical (necessario per Deep Learning Toolbox)
Y_cat = categorical(y_vec, [0, 1], {'Jab', 'Montante'});

%% 2. CHECK DIMENSIONI
sample_1 = XCell{1}; 
num_features  = size(sample_1, 1); 
num_timesteps = size(sample_1, 2); 
fprintf('Input Rete: %d Canali x %d Tempo (fissi).\n', num_features, num_timesteps);

% Controllo Sicurezza
for i = 2:length(XCell)
    if size(XCell{i}, 2) ~= num_timesteps
        error('ERRORE FILE %d: Lunghezza diversa! Rilancia Preprocessing.', i);
    end
end

%% 3. SPLIT DATASET MANUALE (70% Train - 30% Test)

fprintf('--- DIVISIONE DATASET (Manuale) ---\n');

num_samples = length(Y_cat);
split_pct   = 0.70; % Percentuale Training
num_train   = floor(split_pct * num_samples);

% Generazione indici casuali per mescolare i dati
rng(42);
idx_perm = randperm(num_samples);

% Selezione indici
idx_train = idx_perm(1:num_train);
idx_test  = idx_perm(num_train+1:end);

% Creazione Dataset
XTrain = XCell(idx_train);
YTrain = Y_cat(idx_train);
XTest  = XCell(idx_test);
YTest  = Y_cat(idx_test);

fprintf('Dataset diviso: %d Training, %d Test.\n', numel(YTrain), numel(YTest));

%% 4. ARCHITETTURA RETE CNN
layers = [
    sequenceInputLayer(num_features, 'Name', 'input', 'MinLength', num_timesteps, 'Normalization', 'none')
    
    convolution1dLayer(5, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(2, 'Stride', 2)
    
    convolution1dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(2, 'Stride', 2)
    
    globalAveragePooling1dLayer
    
    fullyConnectedLayer(32)
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', 40, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.001, ...
    'ValidationData', {XTest, YTest}, ...
    'ValidationFrequency', 10, ...
    'Plots', 'training-progress', ...
    'Verbose', false, ...
    'Shuffle', 'every-epoch');

%% 5. ADDESTRAMENTO
fprintf('Avvio Training...\n');
[net, info] = trainNetwork(XTrain, YTrain, layers, options);

%% 6. VALUTAZIONE, REPORT E MATRICE CONFUSIONE
fprintf('\n--- CALCOLO RISULTATI FINALI ---\n');

% Predizioni sul Test Set
YPred = classify(net, XTest);

% 1. Calcolo Matrice di Confusione
try
    cm = confusionmat(YTest, YPred);
catch
    % Calcolo manuale se confusionmat non esiste
    labels = categories(YTest);
    num_classes = numel(labels);
    cm = zeros(num_classes);
    for i = 1:numel(YTest)
        r = find(labels == YTest(i)); % Riga: Vera classe
        c = find(labels == YPred(i)); % Colonna: Predetta
        cm(r, c) = cm(r, c) + 1;
    end
end

% Calcolo Metriche
precision = diag(cm) ./ sum(cm, 1)';
recall    = diag(cm) ./ sum(cm, 2);
f1_score  = 2 * (precision .* recall) ./ (precision + recall);

% Gestione NaN (Divisione per zero)
precision(isnan(precision)) = 0;
recall(isnan(recall)) = 0;
f1_score(isnan(f1_score)) = 0;

accuracy = sum(diag(cm)) / sum(cm(:));
final_loss = info.TrainingLoss(end);

% 2. GRAFICO MATRICE DI CONFUSIONE
figure('Name', 'Matrice di Confusione', 'Color', 'w');
try
    cmChart = confusionchart(YTest, YPred);
    cmChart.Title = sprintf('Matrice di Confusione (Accuratezza: %.2f%%)', accuracy*100);
    cmChart.RowSummary = 'row-normalized';    
    cmChart.ColumnSummary = 'column-normalized'; 
    cmChart.FontSize = 12;
catch
    % Fallback visuale se confusionchart manca
    heatmap(categories(YTest), categories(YTest), cm);
    title(sprintf('Confusion Matrix (Acc: %.2f%%)', accuracy*100));
    xlabel('Predicted Class'); ylabel('True Class');
end

% 3. STAMPA REPORT PER POWERPOINT
fprintf('\n======================================================\n');
fprintf('   REPORT FINALE PER PRESENTAZIONE (Copia e Incolla)   \n');
fprintf('======================================================\n');
fprintf('DATI:\n');
fprintf(' -> Totale Campioni:      %d\n', total_files);
fprintf(' -> Training Set (80%%):   %d\n', numel(YTrain));
fprintf(' -> Test Set (20%%):       %d\n', numel(YTest));
fprintf('------------------------------------------------------\n');
fprintf('RISULTATI GLOBALI:\n');
fprintf(' -> Accuratezza Finale:   %.2f %%\n', accuracy * 100);
fprintf(' -> Loss Finale:          %.4f\n', final_loss);
fprintf('------------------------------------------------------\n');
fprintf('METRICHE PER CLASSE:\n');
fprintf('\n   JAB:\n');
fprintf('    - Precision: %.2f %%\n', precision(1) * 100);
fprintf('    - Recall:    %.2f %%\n', recall(1) * 100);
fprintf('    - F1 SCORE:  %.2f %%\n', f1_score(1) * 100);
fprintf('\n   MONTANTE:\n');
fprintf('    - Precision: %.2f %%\n', precision(2) * 100);
fprintf('    - Recall:    %.2f %%\n', recall(2) * 100);
fprintf('    - F1 SCORE:  %.2f %%\n', f1_score(2) * 100);
fprintf('======================================================\n');

% Salvataggio
save('Rete_CNN_Finale.mat', 'net');
fprintf('\nSALVATO: Rete_CNN_Finale.mat (Pronto per Simulink)\n');