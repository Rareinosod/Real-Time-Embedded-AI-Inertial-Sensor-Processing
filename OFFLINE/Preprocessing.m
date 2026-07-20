clc; clear; close all;

%% 1. CONFIGURAZIONE
BASE_INPUT_FOLDER  = '/Users/alessandro/Desktop/AUTOMAZIONE 2/DATASET_GREZZO'; 
BASE_OUTPUT_FOLDER = '/Users/alessandro/Desktop/AUTOMAZIONE 2/DATASET_PROCESSATO_FINALE';

% --- PARAMETRI FILTRO ---
FS_TARGET = 200.0;
LOW_CUT   = 0.5;
HIGH_CUT  = 20;
ALPHA_SMOOTH = 0.2;

% --- PARAMETRI FINESTRA ---
PRE_BUFFER_SEC  = 1.0;
POST_BUFFER_SEC = 1.0;

% --- PARAMETRI TRIGGER ---
K_SOGLIA = 3;
MIN_DIST = 3.0;
WARMUP   = 3.0;

TASKS = {'Jab', 'Montante'}; 

% Creazione cartella output principale
if ~exist(BASE_OUTPUT_FOLDER, 'dir')
    mkdir(BASE_OUTPUT_FOLDER);
    fprintf('Creata cartella output: %s\n', BASE_OUTPUT_FOLDER);
end

%% 2. LOOP PRINCIPALE
for t = 1:length(TASKS)
    currentTask = TASKS{t};
    inputDir = fullfile(BASE_INPUT_FOLDER, currentTask);
    
    if ~isfolder(inputDir)
        warning('Cartella non trovata: %s', inputDir);
        continue;
    end
    
    taskOutDir = fullfile(BASE_OUTPUT_FOLDER, currentTask);
    if ~exist(taskOutDir, 'dir'), mkdir(taskOutDir); end
    
    fprintf('\n==================================================\n');
    fprintf(' ELABORAZIONE: %s (Finestra fissa: -%.1fs / +%.1fs)\n', ...
        upper(currentTask), PRE_BUFFER_SEC, POST_BUFFER_SEC);
    fprintf('==================================================\n');
    
    files = dir(fullfile(inputDir, '*.mat'));
    
    for k = 1:length(files)
        fileName = files(k).name;
        filePath = fullfile(inputDir, fileName);
        [~, subjectName, ~] = fileparts(fileName);
        
        fprintf('-> %s ... ', fileName);
        
        %% A. CARICAMENTO DATI
        try
            raw = load(filePath);
            if isfield(raw, 'datiSensori') && isfield(raw.datiSensori, 'signals')
                valori = raw.datiSensori.signals.values;
                time   = raw.datiSensori.time;
            elseif isfield(raw, 'signals')
                valori = raw.signals.values;
                time   = (0:size(valori,1)-1)'/FS_TARGET; 
            else
                f = fieldnames(raw);
                valori = raw.(f{1});
                time   = (0:size(valori,1)-1)'/FS_TARGET;
            end
        catch
            fprintf('[ERR LETTURA]\n');
            continue;
        end
        
        acc = valori(:, 1:3);
        gyr = valori(:, 4:6);
        
        %% Fs
        Fs = 1/mean(diff(time));
        if isnan(Fs) || isinf(Fs)
            Fs = FS_TARGET;
        end
        
        %% B. FILTRAGGIO
        [b, a] = butter(2, [LOW_CUT HIGH_CUT]/(Fs/2), 'bandpass');
        acc_filt = filtfilt(b, a, acc);
        gyr_filt = filtfilt(b, a, gyr);
        
        acc_smooth = filter(ALPHA_SMOOTH, [1 -(1-ALPHA_SMOOTH)], acc_filt);
        gyr_smooth = filter(ALPHA_SMOOTH, [1 -(1-ALPHA_SMOOTH)], gyr_filt);
        
        %% C. TRIGGER (ACC Y)
        sigTrigger = acc_smooth(:, 2);
        
        [winStart, ~] = detectWindows_EMA( ...
            sigTrigger, Fs, 0.001, K_SOGLIA, MIN_DIST, 0.08, WARMUP);
        
        if isempty(winStart)
            fprintf('0 colpi trovati.\n');
            continue;
        end
        
        %% D. OUTPUT SOGGETTO
        subjDir = fullfile(taskOutDir, subjectName);
        outData = fullfile(subjDir, 'dati');
        if ~exist(outData, 'dir'), mkdir(outData); end
        
        sampPre  = round(PRE_BUFFER_SEC  * Fs);
        sampPost = round(POST_BUFFER_SEC * Fs);
        
        countSaved = 0;
        
        for i = 1:length(winStart)
            triggerIdx = winStart(i);
            
            idx_s = triggerIdx - sampPre;
            idx_e = triggerIdx + sampPost;
            
            if idx_s < 1 || idx_e > length(time)
                continue;
            end
            
            acc_win = acc_smooth(idx_s:idx_e, :);
            gyr_win = gyr_smooth(idx_s:idx_e, :);
            
            %% E. NORMALIZZAZIONE Z-SCORE
            acc_norm = (acc_win - mean(acc_win)) ./ std(acc_win);
            gyr_norm = (gyr_win - mean(gyr_win)) ./ std(gyr_win);
            
            acc_norm(isnan(acc_norm)) = 0;
            gyr_norm(isnan(gyr_norm)) = 0;
            
            %% F. SALVATAGGIO
            finestra.type     = currentTask;
            finestra.acc_norm = acc_norm;
            finestra.gyr_norm = gyr_norm;
            
            save(fullfile(outData, sprintf('colpo_%d.mat', i)), 'finestra');
            
            countSaved = countSaved + 1;
        end
        
        fprintf('%d salvati.\n', countSaved);
    end
end

disp('=== ELABORAZIONE COMPLETATA ===');

%% FUNZIONE TRIGGER EMA
function [winStart, winEnd] = detectWindows_EMA(signal, Fs, alpha, K, minDist, tolUnder, warmup)
    N = numel(signal);
    minDist_S  = round(minDist*Fs);
    tolUnder_S = max(1, round(tolUnder*Fs));
    warmup_S   = round(warmup*Fs);
    
    winStart = [];
    winEnd   = [];
    
    baseline = 0;
    lastEnd  = -inf;
    state    = 0;
    under    = 0;
    startIx  = -1;
    
    abs_sig = abs(signal);
    
    for n = 1:N
        x = abs_sig(n);
        baseline = (1-alpha)*baseline + alpha*x;
        th = K * baseline;
        
        if n <= warmup_S
            continue;
        end
        
        switch state
            case 0
                if x > th && (n - lastEnd) >= minDist_S
                    state = 1;
                    startIx = n;
                    under = 0;
                end
                
            case 1
                if x >= th
                    under = 0;
                else
                    under = under + 1;
                    if under >= tolUnder_S
                        winStart(end+1) = startIx;
                        winEnd(end+1)   = n - under;
                        lastEnd = n - under;
                        state = 0;
                    end
                end
        end
    end
    
    if state == 1
        winStart(end+1) = startIx;
        winEnd(end+1)   = N;
    end
end
