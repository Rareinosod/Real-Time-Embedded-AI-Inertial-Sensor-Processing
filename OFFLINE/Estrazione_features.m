clear; clc; close all;

%% 1. CONFIGURAZIONE PERCORSI
BASE_DATI_FOLDER = '/Users/alessandro/Desktop/AUTOMAZIONE 2/OFFLINE/DATASET_PROCESSATO_FINALE';

OUTPUT_FOLDER = '/Users/alessandro/Desktop/AUTOMAZIONE 2/MATLAB_SVM_FEATURES';
OUTPUT_FILENAME = 'Dataset_Finale_Features.mat';

if ~exist(OUTPUT_FOLDER, 'dir'), mkdir(OUTPUT_FOLDER); end

%% 2. DEFINIZIONE FEATURE E NOMI COLONNE
NOME_FEATURE = {'Mean','StdDev','Variance','Skewness','Kurtosis', ...
                'SumSquares','RMS','Max','Min','Range'};
NUM_FEATURE_PER_ASSE = numel(NOME_FEATURE);

SENSORI = {'ACC','ACC','ACC','GYR','GYR','GYR'};
ASSI    = {'X','Y','Z','X','Y','Z'};
NUM_ASSI = 6; 
NUM_TOT_FEATURE = NUM_FEATURE_PER_ASSE * NUM_ASSI;

Nomi_Colonne_Feature = {};
for i_asse = 1:NUM_ASSI
    sensore = SENSORI{i_asse};
    asse    = ASSI{i_asse};
    for i_f = 1:NUM_FEATURE_PER_ASSE
        nomeFeat = NOME_FEATURE{i_f};
        Nomi_Colonne_Feature{end+1} = sprintf('%s_%s%s', nomeFeat, sensore, asse); 
    end
end

Nomi_Colonne_Dataset = [Nomi_Colonne_Feature, {'Etichetta_Pugno'}]; 

%% 3. LOOP DI ESTRAZIONE
TASKS = {'Jab', 'Montante'};
LABELS = [0, 1]; % 0 = Jab, 1 = Montante

Matrice_Dataset_Finale = []; 

fprintf('Avvio estrazione feature da: %s\n', BASE_DATI_FOLDER);

for t = 1:length(TASKS)
    currentTask = TASKS{t};
    currentLabel = LABELS(t);
    
    taskFolder = fullfile(BASE_DATI_FOLDER, currentTask);
    
    % Ottieni lista soggetti
    subjects = dir(taskFolder);
    subjects = subjects([subjects.isdir]);
    subjects = subjects(~ismember({subjects.name}, {'.', '..'}));
    
    for s = 1:length(subjects)
        subjName = subjects(s).name;
        
        % Percorso: .../Jab/Soggetto/dati/
        dataDir = fullfile(taskFolder, subjName, 'dati');
        
        if ~isfolder(dataDir)
            continue; 
        end
        
        % Lista file colpi
        files = dir(fullfile(dataDir, '*.mat'));
        nColpi = length(files);
        
        fprintf('Processing %s - %s (%d colpi)... ', currentTask, subjName, nColpi);
        
        % Matrice temporanea per questo soggetto
        Matrice_Soggetto = zeros(nColpi, NUM_TOT_FEATURE);
        
        for k = 1:nColpi
            % Caricamento file singolo
            filePath = fullfile(dataDir, files(k).name);
            tmp = load(filePath); 
            
            % Verifica struttura
            if isfield(tmp, 'finestra')
                dati = tmp.finestra; % Struct con acc_norm, gyr_norm
                
                % Estrazione ordinata per i 6 assi
                
                % Vettori dei 6 segnali
                segnali = { ...
                    dati.acc_norm(:, 1), ... % ACC X
                    dati.acc_norm(:, 2), ... % ACC Y
                    dati.acc_norm(:, 3), ... % ACC Z
                    dati.gyr_norm(:, 1), ... % GYR X
                    dati.gyr_norm(:, 2), ... % GYR Y
                    dati.gyr_norm(:, 3)      % GYR Z
                };
                
                col_idx = 1;
                % Per ogni asse, calcola le 10 feature
                for i_asse = 1:NUM_ASSI
                    feats = FunzEstraz_Features_definitiva(segnali{i_asse});
                    
                    % Riempie la riga della matrice
                    for i_f = 1:NUM_FEATURE_PER_ASSE
                         nomeFeat = NOME_FEATURE{i_f};
                         Matrice_Soggetto(k, col_idx) = feats.(nomeFeat);
                         col_idx = col_idx + 1;
                    end
                end
            end
        end
        
        % Aggiungi colonna etichetta (0 o 1) a tutti i colpi del soggetto
        Colonna_Label = ones(nColpi, 1) * currentLabel;
        Blocco_Soggetto = [Matrice_Soggetto, Colonna_Label];
        
        % Accoda alla matrice generale
        Matrice_Dataset_Finale = [Matrice_Dataset_Finale; Blocco_Soggetto];
        fprintf('Fatto.\n');
    end
end

%% 4. SALVATAGGIO
Dataset_Tabella = array2table(Matrice_Dataset_Finale, 'VariableNames', Nomi_Colonne_Dataset);

percorso_salvataggio = fullfile(OUTPUT_FOLDER, OUTPUT_FILENAME);
save(percorso_salvataggio, 'Matrice_Dataset_Finale', 'Dataset_Tabella', 'Nomi_Colonne_Dataset');

fprintf('\n ESTRAZIONE COMPLETATA.\n');
fprintf('File salvato in: %s\n', percorso_salvataggio);
fprintf('Totale campioni estratti: %d\n', size(Matrice_Dataset_Finale, 1));


function features = FunzEstraz_Features_definitiva(data)
% Estrae 10 feature dal segnale
    
    data = data(:);
    N = length(data);
    features = struct();

    if N == 0
        features = initEmptyFeatures();
        return;
    end

    % 1) Statistiche base
    features.Mean      = mean(data);
    features.StdDev    = std(data);
    features.Variance  = var(data);
    features.Max       = max(data);
    features.Min       = min(data);
    features.Range     = features.Max - features.Min;

    % 2) Forma distribuzione
    features.Skewness  = skewness(data);
    features.Kurtosis  = kurtosis(data);

    % 3) Energia
    features.SumSquares = sum(data.^2);
    features.RMS        = rms(data);
end

function features = initEmptyFeatures()
    names = {'Mean','StdDev','Variance','Skewness','Kurtosis', ...
             'SumSquares','RMS','Max','Min','Range'};
    for i = 1:numel(names)
        features.(names{i}) = 0;
    end
end