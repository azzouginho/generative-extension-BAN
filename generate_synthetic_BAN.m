% Generative Model for Brain Artery Networks (BAN)
% Method: Tangent Space PCA + Topological Bootstrapping
% Output: 
%   1. 3D Visualizations (Clean style: No spheres, consistent edge colors)
%   2. Statistical Validation (Histograms N=92 vs N=92 + Train Mean)

clear; close all;
addPath; % Ensure library functions (matchEG, etc.) are in the path

% --- PARAMETERS ---
DATA_DIR    = './data/BrainArteryTree/'; 
CACHE_DIR   = 'cache_results';          
RESULTS_DIR = 'results_figures';      
N_LEARN     = 13;   % Number of subjects for PCA training (Subset)
N_GEN_STATS = 92;   % Number of synthetic subjects for validation stats
N_VISU      = 3;    % Number of 3D examples to display
COMPONENTS  = {'Left', 'Right', 'Top', 'Bottom'}; 

if ~exist(CACHE_DIR, 'dir'), mkdir(CACHE_DIR); end
if ~exist(RESULTS_DIR, 'dir'), mkdir(RESULTS_DIR); end

%% MAIN PROCESSING LOOP
for c = 1:length(COMPONENTS)
    comp_name = COMPONENTS{c};
    fprintf('\n=== Processing Component: %s ===\n', comp_name);
    
    % --- 1. LOAD FULL RAW DATA ---
    raw_file = fullfile(DATA_DIR, sprintf('BrainTree%s.mat', comp_name));
    if ~exist(raw_file, 'file')
        warning('File not found: %s', raw_file); 
        continue; 
    end
    
    tmp = load(raw_file); 
    names = fieldnames(tmp); 
    DataFull = tmp.(names{1}); 
    n_total_real = length(DataFull);
    
    % --- 2. COMPUTE GROUND TRUTH STATISTICS (N=92) ---
    fprintf('-> Computing descriptors on ALL %d Real subjects...\n', n_total_real);
    Real_AvgLen = []; Real_Curv = [];
    
    for i = 1:n_total_real
        st = compute_descriptors(DataFull(i));
        Real_AvgLen = [Real_AvgLen; st.AvgLength];
        Real_Curv   = [Real_Curv; st.AvgCurvature];
    end
    
    % --- 3. COMPUTE TRAINING SUBSET STATS (N=13) ---
    n_learn_actual = min(N_LEARN, n_total_real);
    DataLearn = DataFull(1:n_learn_actual);
    
    Train_AvgLen = []; Train_Curv = [];
    for i = 1:n_learn_actual
        st = compute_descriptors(DataLearn(i));
        Train_AvgLen = [Train_AvgLen; st.AvgLength];
        Train_Curv   = [Train_Curv; st.AvgCurvature];
    end
    
    % Mean values for vertical lines in histograms
    mu_train_len = mean(Train_AvgLen);
    mu_train_curv = mean(Train_Curv);
    
    % --- 4. TRAIN GENERATIVE MODEL (or Load from Cache) ---
    final_cache_file = fullfile(CACHE_DIR, sprintf('Matched_%s_N%d.mat', comp_name, N_LEARN));
    
    if exist(final_cache_file, 'file') == 2
        disp('-> Loading model from cache...');
        loaded = load(final_cache_file);
        MatchedEG = loaded.MatchedEG;
        MeanEG    = loaded.MeanEG;
        Mean_q    = loaded.Mean_q;
        U = loaded.U; S = loaded.S;
    else
        disp('-> Training model (Matching + PCA)...');
        
        % A. Group Alignment
        n_nodes_list = arrayfun(@(x) size(x.A,1), DataLearn);
        [~, idx_template] = max(n_nodes_list);
        Template = DataLearn(idx_template);
        
        % Parallel Matching
        temp_dir = fullfile(CACHE_DIR, ['temp_' comp_name]);
        if ~exist(temp_dir, 'dir'), mkdir(temp_dir); end
        if isempty(gcp('nocreate')), parpool; end
        
        parfor i = 1:n_learn_actual
            fname = fullfile(temp_dir, sprintf('subj_%d.mat', i));
            if exist(fname, 'file'), continue; end
            if i == idx_template, Subj = DataLearn(i); else, [Subj, ~, ~, ~] = matchEG(DataLearn(i), Template); end
            parsave(fname, Subj);
        end
        
        MatchedEG = DataLearn;
        for i = 1:n_learn_actual, t = load(fullfile(temp_dir, sprintf('subj_%d.mat', i))); MatchedEG(i) = t.data; end
        rmdir(temp_dir, 's');
        
        % B. Tangent PCA
        MeanEG = avgEG(MatchedEG, false, true);
        [Mean_q, ~] = adjbeta2q(MeanEG.Abeta, MeanEG.A);
        num_el = numel(Mean_q);
        Vecs = zeros(num_el, n_learn_actual);
        for i = 1:n_learn_actual
            [Sq, ~] = adjbeta2q(MatchedEG(i).Abeta, MatchedEG(i).A);
            Vecs(:,i) = Sq(:) - Mean_q(:);
        end
        [U, S, ~] = svd(Vecs, 'econ');
        eigenvalues = diag(S).^2 / (n_learn_actual - 1);
        
        save(final_cache_file, 'MatchedEG', 'MeanEG', 'Mean_q', 'U', 'S', 'n_learn_actual', 'eigenvalues');
    end

    % --- 5. GENERATE DATA FOR STATISTICS (N=92 Random Samples) ---
    fprintf('-> Generating %d Synthetic subjects for stats...\n', N_GEN_STATS);
    Synth_AvgLen = []; Synth_Curv = [];
    
    for g = 1:N_GEN_STATS
        % Geometry Sampling
        coeffs = randn(n_learn_actual, 1);
        v_syn = U * (S * coeffs / sqrt(n_learn_actual-1));
        q_full = reshape(Mean_q(:) + v_syn, size(Mean_q));
        
        % Topology Bootstrap (Random)
        rand_idx = randi(n_learn_actual);
        TopoMask = MatchedEG(rand_idx).A;
        NodesRef = MatchedEG(rand_idx).nodeXY;
        
        % Reconstruction & Anchoring
        Abeta_Raw = adjq2beta(q_full, TopoMask);
        SynEG = MeanEG; SynEG.A = TopoMask; SynEG.nodeXY = NodesRef;
        
        [n_nodes, ~] = size(TopoMask);
        Abeta_Final = zeros(size(Abeta_Raw));
        for r = 1:n_nodes
            for c = r+1:n_nodes
                if SynEG.A(r,c) > 0
                    raw_c = Abeta_Raw(:,:,r,c);
                    pA = NodesRef(:,r); pB = NodesRef(:,c);
                    aligned = shape2curve(raw_c, pA, pB);
                    aligned(:,1) = pA; aligned(:,end) = pB; 
                    Abeta_Final(:,:,r,c) = aligned; Abeta_Final(:,:,c,r) = fliplr(aligned);
                end
            end
        end
        SynEG.Abeta = Abeta_Final; SynEG.beta = adj2beta(Abeta_Final, SynEG.A);
        
        % Compute Stats
        st = compute_descriptors(SynEG);
        Synth_AvgLen = [Synth_AvgLen; st.AvgLength];
        Synth_Curv   = [Synth_Curv; st.AvgCurvature];
    end
    
    % --- 6. PLOT 1: 3D VISUALIZATION (USER'S STYLE - NO NODES) ---
    % Logic: Pick ONE template, show it, then generate 3 synthetic samples based on THAT template.
    
    h_fig = figure('Name', ['3D Examples: ' comp_name], 'Color', 'w', 'Position', [50 500 1600 400]);
    
    % Pick random template for visualization
    tpl_idx = randi(n_learn_actual);
    TemplatePatient = MatchedEG(tpl_idx);
    TemplateMask = TemplatePatient.A;
    TemplateNodes = TemplatePatient.nodeXY;
    
    % Setup Colors (One color per edge index)
    [row_idx, col_idx] = find(triu(TemplateMask,1));
    n_edges = length(row_idx);
    cmap = lines(n_edges);
    [n_nodes_tpl, ~] = size(TemplateMask);

    % A. Display Template
    subplot(1, N_VISU+1, 1);
    hold on;
    for e = 1:n_edges
        r = row_idx(e); c = col_idx(e);
        curve = TemplatePatient.Abeta(:,:,r,c);
        plot3(curve(1,:), curve(2,:), curve(3,:), 'Color', cmap(e,:), 'LineWidth', 1.5);
    end
    title('Real Template', 'FontWeight', 'bold');
    axis equal off; view(3); camlight;
    hold off;
    
    % B. Generate & Display Synthetic (Fixed Topology)
    for g = 1:N_VISU
        % Sampling
        coeffs = randn(n_learn_actual, 1);
        v_syn = U * (S * coeffs / sqrt(n_learn_actual-1));
        q_full = reshape(Mean_q(:) + v_syn, size(Mean_q));
        
        % Reconstruction (Using the VISU Template topology)
        Abeta_Raw = adjq2beta(q_full, TemplateMask);
        SynEG = MeanEG; 
        SynEG.A = TemplateMask; 
        SynEG.nodeXY = TemplateNodes;
        
        Abeta_Final = zeros(size(Abeta_Raw));
        
        % Correction Loop
        for r = 1:n_nodes_tpl
            for c = r+1:n_nodes_tpl
                if SynEG.A(r,c) > 0
                    raw_c = Abeta_Raw(:,:,r,c);
                    pA = TemplateNodes(:,r); pB = TemplateNodes(:,c);
                    
                    aligned = shape2curve(raw_c, pA, pB);
                    aligned(:,1) = pA; aligned(:,end) = pB; % Force endpoints
                    
                    Abeta_Final(:,:,r,c) = aligned; 
                    Abeta_Final(:,:,c,r) = fliplr(aligned);
                end
            end
        end
        SynEG.Abeta = Abeta_Final;
        
        % Plotting
        subplot(1, N_VISU+1, g+1);
        hold on;
        for e = 1:n_edges
            r = row_idx(e); c = col_idx(e);
            curve = SynEG.Abeta(:,:,r,c);
            plot3(curve(1,:), curve(2,:), curve(3,:), 'Color', cmap(e,:), 'LineWidth', 1.5);
        end
        title(['Synthetic #' num2str(g)], 'Color', 'r', 'FontWeight', 'bold');
        axis equal off; view(3); camlight;
        hold off;
    end
    saveas(h_fig, fullfile(RESULTS_DIR, ['Examples3D_' comp_name '.png']));
    
    % --- 7. PLOT 2: STATISTICAL VALIDATION (HISTOGRAMS) ---
    figHist = figure('Name', ['Statistics: ' comp_name], 'Color', 'w', 'Position', [50 100 1000 400]);
    nbins = 20; 
    
    % A. Average Artery Length
    subplot(1, 2, 1);
    all_data = [Real_AvgLen; Synth_AvgLen];
    bin_edges = linspace(min(all_data), max(all_data), nbins+1);
    
    hold on;
    histogram(Real_AvgLen, 'BinEdges', bin_edges, 'Normalization', 'probability', 'FaceColor', 'b', 'FaceAlpha', 0.4);
    histogram(Synth_AvgLen, 'BinEdges', bin_edges, 'Normalization', 'probability', 'FaceColor', 'r', 'FaceAlpha', 0.5);
    xline(mu_train_len, '--g', 'LineWidth', 2.5);
    
    title('Average Artery Length', 'FontWeight', 'bold'); 
    ylabel('Probability'); 
    legend(['Full Real (N=' num2str(n_total_real) ')'], ...
           ['Synth (N=' num2str(N_GEN_STATS) ')'], ...
           ['Train Mean (N=' num2str(n_learn_actual) ')']); 
    grid on;
    
    % B. Average Curvature
    subplot(1, 2, 2);
    all_data = [Real_Curv; Synth_Curv];
    bin_edges = linspace(min(all_data), max(all_data), nbins+1);
    
    hold on;
    histogram(Real_Curv, 'BinEdges', bin_edges, 'Normalization', 'probability', 'FaceColor', 'b', 'FaceAlpha', 0.4);
    histogram(Synth_Curv, 'BinEdges', bin_edges, 'Normalization', 'probability', 'FaceColor', 'r', 'FaceAlpha', 0.5);
    xline(mu_train_curv, '--g', 'LineWidth', 2.5);
    
    title('Average Curvature', 'FontWeight', 'bold'); 
    ylabel('Probability'); 
    legend('Full Real', 'Synth', 'Train Mean'); 
    grid on;
    
    sgtitle(['Distribution Comparison: ' comp_name], 'FontWeight', 'bold');
    
    saveas(figHist, fullfile(RESULTS_DIR, ['Histograms_' comp_name '.png']));
    fprintf('-> Figures saved in %s\n', RESULTS_DIR);
end

%% HELPER FUNCTIONS

function parsave(fname, data)
    save(fname, 'data');
end

function stats = compute_descriptors(EG)
    [n, ~] = size(EG.A);
    lens = []; curvs = [];
    for i = 1:n
        for j = i+1:n
            if EG.A(i,j) > 0
                c = EG.Abeta(:,:,i,j); 
                d = diff(c, 1, 2);
                seg_l = sqrt(sum(d.^2, 1));
                L = sum(seg_l);
                if L < 1e-6, continue; end
                
                tangents = d ./ seg_l;
                dp = dot(tangents(:, 1:end-1), tangents(:, 2:end));
                dp = max(min(dp, 1), -1); 
                K = sum(acos(dp));
                lens = [lens, L]; curvs = [curvs, K];
            end
        end
    end
    stats.AvgLength = mean(lens);
    stats.AvgCurvature = mean(curvs);
end