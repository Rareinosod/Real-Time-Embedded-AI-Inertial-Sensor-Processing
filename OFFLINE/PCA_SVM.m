clear; clc; close all;

%% 1) CARICAMENTO DEL DATASET
matFile = 'prove/Dataset_Matrice_UNICA/Dataset_Finale_Features_definitivo.mat'; 
if exist(matFile, 'file') ~= 2
    error('File .mat non trovato: %s', matFile);
end
data = load(matFile);
fields = fieldnames(data);
raw = data.(fields{1});   

if istable(raw)
    X = raw{:,1:end-1};           
    y = raw{:,end};               
    featureNames = raw.Properties.VariableNames(1:end-1);
else
    X = raw(:,1:end-1);
    y = raw(:,end);
    featureNames = [];
end

%% 2) TRAIN/TEST SPLIT (70/30) 
rng(1); % Per riproducibilità
cv = cvpartition(y, 'HoldOut', 0.3);
idxTrain = training(cv);
idxTest  = test(cv);

X_train_raw = X(idxTrain, :);
y_train     = y(idxTrain);
X_test_raw  = X(idxTest, :);
y_test      = y(idxTest);

fprintf('Dataset: %d Train, %d Test\n', sum(idxTrain), sum(idxTest));

%% 3) STANDARDIZZAZIONE 
mu_train    = mean(X_train_raw);
sigma_train = std(X_train_raw);

% Trasformazione Z-score manuale
Z_train = (X_train_raw - mu_train) ./ (sigma_train + eps); 
Z_test  = (X_test_raw - mu_train) ./ (sigma_train + eps);

%% 4) PCA (Calcolata solo sul Training Set)
nSamplesTrain = size(Z_train, 1);

%Calcolo matrice di Covarianza
Sigma_train = (1/(nSamplesTrain - 1)) * (Z_train' * Z_train);

%Calcolo degli Autovettori e Auotovalori
[Phi, D] = eig(Sigma_train);
lambda = diag(D);
[lambdaSorted, idx] = sort(lambda, 'descend');
Phi_sorted = Phi(:, idx);

% Varianza Spiegata
totalVar = sum(lambdaSorted);
explained = 100 * (lambdaSorted / totalVar);
cumExplained = cumsum(explained);

% Scelta numero componenti (es. 95% varianza)
sogliaVar = 95;
k = find(cumExplained >= sogliaVar, 1, 'first');
W = Phi_sorted(:, 1:k); 

fprintf('PCA: Selezionate %d componenti per spiegare il %.1f%% della varianza.\n', k, sogliaVar);

%% 5) PROIEZIONE NELLO SPAZIO PCA
X_train_pca = Z_train * W;
X_test_pca  = Z_test * W;

%% 6) TRAINING SVM LINEARE
% Usiamo 'Standardize', false perché abbiamo già standardizzato manualmente
SVMModel = fitcsvm(X_train_pca, y_train, ...
    'KernelFunction', 'linear', ...
    'Standardize', false, ... 
    'ClassNames', [0, 1]);

%% 7) TEST E METRICHE DETTAGLIATE
[y_pred, ~] = predict(SVMModel, X_test_pca);

N  = numel(y_test);
tp = sum((y_test == 1) & (y_pred == 1));   % montante corretti
tn = sum((y_test == 0) & (y_pred == 0));   % jab corretti
fp = sum((y_test == 0) & (y_pred == 1));   % jab classificati come montante
fn = sum((y_test == 1) & (y_pred == 0));   % montante classificati come jab

accuracy  = (tp + tn) / N;
precision = tp / (tp + fp + eps);
recall    = tp / (tp + fn + eps);
f1        = 2 * precision * recall / (precision + recall + eps);
loss      = 1 - accuracy;

fprintf('\n=== Metriche sul TEST set (Senza Data Leakage) ===\n');
fprintf('Accuracy   = %.4f (%.2f%%)\n', accuracy, accuracy*100);
fprintf('Precision  = %.4f\n', precision);
fprintf('Recall     = %.4f\n', recall);
fprintf('F1-score   = %.4f\n', f1);
fprintf('Loss       = %.4f\n', loss);


%% %% 8) MATRICE DI CONFUSIONE
y_test_c = categorical(y_test, [0 1], {'JAB','MONTANTE'});
y_pred_c = categorical(y_pred, [0 1], {'JAB','MONTANTE'});


f = figure('Name', 'Validazione Modello', 'Color', 'w'); 

% Creazione della matrice di confusione
cm = confusionchart(y_test_c, y_pred_c, 'Normalization','absolute');


cm.Title = 'Matrice di Confusione SVM';
cm.FontName = 'Arial';
cm.FontSize = 22;


cm.RowSummary = 'off';
cm.ColumnSummary = 'off';
%% 9) VISUALIZZAZIONE SVM (Prime 2 Componenti)
if k >= 2
    figure('Color', 'w');
    hold on; grid on;
    
    % Plot punti di train
    gscatter(X_train_pca(:,1), X_train_pca(:,2), y_train, 'br', 'o', 14);
    
    % Plot Support Vectors
    sv = SVMModel.SupportVectors;
    plot(sv(:,1), sv(:,2), 'ko', 'MarkerSize', 14, 'LineWidth', 1.5);
    %ylim([0, 8])
    xlabel('PC1', 'FontSize', 22, 'FontWeight', 'bold'); 
    ylabel('PC2', 'FontSize', 22, 'FontWeight', 'bold');
    
    title('SVM Lineare: Spazio PC1-PC2', 'FontSize',22);
    legend('JAB (Train)','MONTANTE (Train)','Support vectors','Location','best');
    lgd.FontSize = 22;
    ax=gca;
    ax.FontSize=22;
    
end

%% 10) SALVATAGGIO PER SIMULINK / ONLINE
SVMcompact = compact(SVMModel);
mu = mu_train;
sigma = sigma_train;

save('modello_PCA_SVM_VALIDO.mat', 'mu', 'sigma', 'W', 'SVMcompact');
fprintf('\nParametri salvati in "modello_PCA_SVM_VALIDO.mat".\n');
