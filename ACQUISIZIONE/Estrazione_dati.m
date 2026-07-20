%% 1. ESTRAZIONE E PLOT
% Accedi alla struttura salvata
stutturaDati = datiSensori; 

% Estrai il vettore del tempo
tempo = stutturaDati.time;

% Estrai i valori
valori = stutturaDati.signals.values;

% Separa i canali
ax = valori(:, 1);
ay = valori(:, 2);
az = valori(:, 3);
gx = valori(:, 4);
gy = valori(:, 5);
gz = valori(:, 6);

% Plot rapido
figure;
subplot(2,1,1); plot(tempo, [gx, gy, gz]); 
title('Giroscopio'); legend('X','Y','Z'); grid on;
subplot(2,1,2); plot(tempo, [ax, ay, az]); 
title('Accelerometro'); legend('X','Y','Z'); grid on;

%% 2. SALVATAGGIO NELLA CARTELLA 'Prove'
% Definisci il nome della sottocartella
nomeCartella = 'Prove';

% Controlla se la cartella esiste già. Se non esiste, la crea.
if ~exist(nomeCartella, 'dir')
    mkdir(nomeCartella);
    fprintf('La cartella "%s" non esisteva ed è stata creata.\n', nomeCartella);
end

% Chiedi il nome del file (senza estensione .mat)
nomeFileUtente = input('Inserisci il nome del file (es. Soggetto1_attività1):', 's');

% Aggiungi l'estensione .mat se non scritta
if ~endsWith(nomeFileUtente, '.mat')
    nomeFileUtente = [nomeFileUtente, '.mat'];
end

% Crea il percorso completo (CurrentFolder/Prove/NomeFile.mat)
percorsoCompleto = fullfile(pwd, nomeCartella, nomeFileUtente);

% Salva SOLO la variabile 'datiSensori' in quel percorso
save(percorsoCompleto, 'datiSensori');

fprintf('Salvataggio completato! File salvato in:\n%s\n', percorsoCompleto);