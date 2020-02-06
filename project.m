%% IMPROTING DATA SET
Nfeatures=22;
Ninstances=195;
Ntr_inst=156;

fid = fopen('excel_parkinsons_without_label.data');
parsed_file = textscan(fid,'%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f','delimiter',',');
fclose(fid);
data = cell2mat(parsed_file(:,:)); %ignore the last column of strings
labels=parsed_file{1};
featlab=[ ...
    'Status          ';
    'MDVP:Fo(Hz)     '; %1
    'MDVP:Fhi(Hz)    '; %2
    'MDVP:Flo(Hz)    '; %3
    'MDVP:Jitter(%)  '; %4
    'MDVP:Jitter(Abs)'; %5
    'MDVP:RAP        '; %6
    'MDVP:PPQ        '; %7
    'Jitter:DDP      '; %8
    'MDVP:Shimmer    '; %9
    'MDVP:Shimmer(dB)'; %10
    'Shimmer:APQ3    '; %12
    'Shimmer:APQ5    '; %12
    'MDVP:APQ        '; %14
    'Shimmer:DDA     '; %14
    'NHR             '; %15
    'HNR             '; %16
    'RPDE            '; %17
    'DFA             '; %18
    'spread1         '; %19
    'spread2         '; %20
    'D2              '; %21
    'PPE             '; %22
];


prdat=prdataset(data,labels,'featlab',featlab);
prdat_without_labels = prdat(:, [2:end]);
%prdat_without_labels(:, [5 6 7 8 10 11 12 13 14 15]) = [];
prdat_struct=struct(prdat_without_labels);

priors = getprior(prdat_without_labels);
prdat_without_labels = setprior(prdat_without_labels, priors);

class1 = getdata(prdat_without_labels, 1);
class2 = getdata(prdat_without_labels, 2);

true_error_qdc = 0;
true_error_ldc = 0;
true_error_nmsc = 0;
true_error_knnc = 0;
true_error_qdc_paper = 0;
true_error_ldc_paper = 0;
true_error_nmsc_paper = 0;
true_error_knnc_paper = 0;

[prtrain, prtest]=gendat(prdat_without_labels, 0.65);

%% CORRELATION MATRIX
covariance_matrix_before = cov(getdata(prdat_without_labels));
correlation_matrix_before = corrcov(covariance_matrix_before);
correlation_matrix_before_image = imagesc(correlation_matrix_before)


%% COMPUTING MEAN ERROR OF ENTIRE SET
n = 100;
for i = 1 : n
    [prtrain, prtest]=gendat(prdat_without_labels, 0.65);
    W1 = qdc(prtrain); W2 = ldc(prtrain); W3 = nmsc(prtrain); [W4,K,E] = knnc(prtrain);
    true_error_qdc = true_error_qdc + testc(prtest * W1);  
    true_error_ldc = true_error_ldc + testc(prtest * W2);
    true_error_nmsc = true_error_nmsc + testc(prtest * W3);
    true_error_knnc = true_error_knnc + testc(prtest * W4);
    if i == n
        true_error_qdc = true_error_qdc/n;
        true_error_ldc = true_error_ldc/n;
        true_error_nmsc = true_error_nmsc/n;
        true_error_knnc = true_error_knnc/n;
        disp('ENTIRE SET: Gendat Error 1000 repitions -- QDC -- LDC -- NMSC -- KNNC ')
        disp([true_error_qdc true_error_ldc true_error_nmsc true_error_knnc])
    end
end

%% COMPUTING MEAN ERROR OF ENTIRE SET - REGULARIZATION
for i = 1 : n
     [prtrain, prtest]=gendat(prdat_without_labels, 0.65);
     W1 = qdc(prtrain, 0, 0.8); W2 = ldc(prtrain, 0, 0.8);
     true_error_qdc = true_error_qdc + testc(prtest * W1);  
     true_error_ldc = true_error_ldc + testc(prtest * W2);
     if i == n
         true_error_qdc = true_error_qdc/n;
         true_error_ldc = true_error_ldc/n;
 
         disp('REGULARIZATION Gendat Error 1000 repitions -- QDC -- LDC')
         disp([true_error_qdc true_error_ldc])
     end
 end


%% BUILDING CLASSIFIERS - ROC CURVE FOR EVERY CLASSIFIER
%Builiding classifiers
W1 = qdc(prtrain); W2 = ldc(prtrain); W3 = nmsc(prtrain); W4 = knnc(prtrain);

% ROC plots                       
r_qdc = prroc(prtest,W1);             % Compute the ROC curve with test data (see pp.40)
r_qdc.error = 1 - r_qdc.error;            % Just to get something similar to Fig 2.13(b)

r_ldc = prroc(prtest,W2);             
r_ldc.error = 1 - r_ldc.error;            

r_nmsc = prroc(prtest,W3);
r_nmsc.error = 1 - r_nmsc.error;

r_knnc = prroc(prtest,W4);
r_knnc.error = 1 - r_knnc.error;

%Confusion Matrix here?

%Plot and input the labels
%Looks like my data is positively correlated, suggesting that they are
%doing similar things
figure(5)
sgtitle('ROC Curves for various classifiers', 'FontWeight', 'Bold')
x0 = 400;
y0 = -50;
width = 1000;
height = 1000;
set(gcf, 'position', [x0, y0, width, height])

subplot(221); plote(r_qdc); title('QDC'); xlabel('P_{fa}'); ylabel('P_{det}'); axis equal tight; grid on;
subplot(222); plote(r_ldc); title('LDC'); xlabel('P_{fa}'); ylabel('P_{det}'); axis equal tight; grid on;
subplot(223); plote(r_nmsc); title('NMSC'); xlabel('P_{fa}'); ylabel('P_{det}'); axis equal tight; grid on;
subplot(224); plote(r_knnc); title('KNNC'); xlabel('P_{fa}'); ylabel('P_{det}'); axis equal tight; grid on;


%% CROSS VALIDATION

% True error
true_error = [testc(prtest * W1) testc(prtest * W2) testc(prtest * W3) testc(prtest * W4)];

%QDC
cross_val_qdc_1 = prcrossval(prdat_without_labels,qdc,10,1);
cross_val_qdc_5 = prcrossval(prdat_without_labels,qdc,10,5);
cross_val_qdc_25 = prcrossval(prdat_without_labels,qdc,10,25);
disp('QDC: ')
disp([true_error(1), true_error_qdc, cross_val_qdc_1, cross_val_qdc_5, cross_val_qdc_25])

%LDC
cross_val_ldc_1 = prcrossval(prdat_without_labels,ldc,10,1);
cross_val_ldc_5 = prcrossval(prdat_without_labels,ldc,10,5);
cross_val_ldc_25 = prcrossval(prdat_without_labels,ldc,10,25);
disp('LDC: ')
disp([true_error(2), true_error_ldc, cross_val_ldc_1, cross_val_ldc_5, cross_val_ldc_25])
 
% %NMSC
cross_val_nmsc_1 = prcrossval(prdat_without_labels,nmsc,10,1);
cross_val_nmsc_5 = prcrossval(prdat_without_labels,nmsc,10,5);
cross_val_nmsc_25 = prcrossval(prdat_without_labels,nmsc,10,25);
disp('NMSC: ')
disp([true_error(3), true_error_nmsc, cross_val_nmsc_1, cross_val_nmsc_5, cross_val_nmsc_25])
 
% %KNNC
cross_val_knnc_1 = prcrossval(prdat_without_labels,knnc,10,1);
cross_val_knnc_5 = prcrossval(prdat_without_labels,knnc,10,5);
cross_val_knnc_25 = prcrossval(prdat_without_labels,knnc,10,25);
disp('KNNC: ')
disp([true_error(4), true_error_knnc, cross_val_knnc_1, cross_val_knnc_5, cross_val_knnc_25])




%% FEATURE SELECTION AND TRAINING OF CLASSIFIERS - CONFUSION MATRIX BEFORE AND AFTER
[prtrain, prtest]=gendat(prdat_without_labels, 0.65);

%True error
W1 = qdc(prtrain); W2 = ldc(prtrain); W3 = nmsc(prtrain); W4 = knnc(prtrain);
true_error = [testc(prtest * W1) testc(prtest * W2) testc(prtest * W3) testc(prtest * W4)];

% Initial measure of seperability using Inter/Intra
initial_seperability_inter_intra = feateval(prdat_without_labels, 'in-in');
initial_seperatbility_NN = feateval(prdat_without_labels, 'NN');
disp('Before Feature Selection Measure of seperability --- Inter/Intra & Nearest-Neighbor: ');
disp([+initial_seperability_inter_intra; +initial_seperatbility_NN]');

% Displaying Error rates and Conf-mat of classifiers BEFORE feature
% selection
disp('Before Feature Selection - QDC Error Rate: ')
disp([true_error(1)])
confmat_all_qdc = (prtest * W1);
confmat(confmat_all_qdc)
disp('--------------------------------------------------------------')


disp('Before Feature Selection - LDC Error Rate: ')
disp([true_error(2)])
confmat_all_ldc = (prtest * W2);
confmat(confmat_all_ldc)
disp('--------------------------------------------------------------')


disp('Before Feature Selection - NMSC Error Rate: ')
disp([true_error(3)])
confmat_all_nmsc = (prtest * W3);
confmat(confmat_all_nmsc)
disp('--------------------------------------------------------------')


disp('Before Feature Selection - KNNC Error Rate: ')
disp([true_error(4)])
confmat_all_knnc = (prtest * W4);
confmat(confmat_all_knnc)
disp('--------------------------------------------------------------')


% Using the Inter-Intra and Nearest-Neighbor method to find the best
% features using Branch-And-Bound
[BB_selection_in_in, BB_in_in_R] = featselm(prdat_without_labels, 'in-in', 'b&b');
[BB_selection_NN, BB_NN_R] = featselm(prdat_without_labels, 'NN', 'b&b');
disp('B&B best features are - Inter/Intra & Nearest-Neighbor:');
disp([+BB_selection_in_in; +BB_selection_NN]');

% Using the Inter-Intra and Nearest-Neighbor method to find the best
% features using Bottom-Up Feature Selection
[Forward_selection_in_in, Forward_in_in_R] = featselm(prdat_without_labels, 'in-in', 'forward', 2);
[Forward_selection_NN, Forward_NN_R] = featselm(prdat_without_labels, 'NN', 'forward', 2);
disp('Forward best features are - Inter/Intra & Nearest-Neighbor:');
disp([+Forward_selection_in_in; +Forward_selection_NN]');

% Computing new J-Value based on Inter/intra and Nearest-Neighbor
post_seperability_inter_intra = feateval(prdat_without_labels(:, [3 19]), 'in-in');
post_seperatbility_NN = feateval(prdat_without_labels(:, [3 19]), 'NN');
disp('After Feature Selection Measure of seperability --- Inter/Intra & Nearest-Neighbor: ');
disp([+post_seperability_inter_intra; +post_seperatbility_NN]');

% Train classifiers with selected features
W1_selected = qdc(prtrain(:, [3 19])); W2_selected = ldc(prtrain(:, [3 19])); W3_selected = nmsc(prtrain(:, [3 19])); W4_selected = knnc(prtrain(:, [3 19]));
true_error_qdc_selected = testc(prtest(:, [3 19]) * W1_selected);  
true_error_ldc_selected = testc(prtest(:, [3 19]) * W2_selected);
true_error_nmsc_selected = testc(prtest(:, [3 19]) * W3_selected);
true_error_knnc_selected = testc(prtest(:, [3 19]) * W4_selected);

% Displaying Error rates and Conf-mat of classifiers AFTER feature
% selection
disp('After Feature Selection - QDC Error Rate: ')
disp([true_error_qdc_selected])
confmat_selected_qdc = (prtest(:, [3 19]) * W1_selected);
confmat(confmat_selected_qdc)
disp('--------------------------------------------------------------')


disp('After Feature Selection - LDC Error Rate: ')
disp([true_error_ldc_selected])
confmat_selected_ldc = (prtest(:, [3 19]) * W2_selected);
confmat(confmat_selected_ldc)
disp('--------------------------------------------------------------')


disp('After Feature Selection - NMSC Error Rate: ')
disp([true_error_nmsc_selected])
confmat_selected_nmsc = (prtest(:, [3 19]) * W3_selected);
confmat(confmat_selected_nmsc)
disp('--------------------------------------------------------------')


disp('After Feature Selection - KNNC Error Rate: ')
disp([true_error_knnc_selected])
confmat_selected_knnc = (prtest(:, [3 19]) * W4_selected);
confmat(confmat_selected_knnc)
disp('--------------------------------------------------------------')


 
figure(101); scatterd(prdat_without_labels(:, [3 19]));
plotc(W1_selected);
plotm(W1_selected, 2);

% figure(102); scatterd(prdat_without_labels(:, [3 19]));
% plotc(W2_selected);
% plotm(W2_selected, 2);


%% BEST FEATURES ACCORDING TO PAPER
n = 100;
for i = 1 : n
    [prtrain, prtest]=gendat(prdat_without_labels, 0.65);
    W1_selected_PAPER = qdc(prtrain(:, [16 17 18 22])); W2_selected_PAPER = ldc(prtrain(:, [16 17 18 22])); W3_selected_PAPER = nmsc(prtrain(:, [16 17 18 22])); W4_selected_PAPER = knnc(prtrain(:, [16 17 18 22]));
    true_error_qdc_paper = true_error_qdc_paper + testc(prtest(:, [16 17 18 22]) * W1_selected_PAPER);  
    true_error_ldc_paper = true_error_ldc_paper + testc(prtest(:, [16 17 18 22]) * W2_selected_PAPER);
    true_error_nmsc_paper = true_error_nmsc_paper + testc(prtest(:, [16 17 18 22]) * W3_selected_PAPER);
    true_error_knnc_paper = true_error_knnc_paper + testc(prtest(:, [16 17 18 22]) * W4_selected_PAPER);
    if i == n
        true_error_qdc_paper = true_error_qdc_paper/n;
        true_error_ldc_paper = true_error_ldc_paper/n;
        true_error_nmsc_paper = true_error_nmsc_paper/n;
        true_error_knnc_paper = true_error_knnc_paper/n;
        disp('FOUR BEST: Gendat Error 100 repitions -- QDC -- LDC -- NMSC -- KNNC ')
        disp([true_error_qdc_paper true_error_ldc_paper true_error_nmsc_paper true_error_knnc_paper])
    end
end

%% FEATURE EXTRACTION & DIMENSIONALITY REDUCTION (PCA, CLUSTERING)
% PCA chooses the subspace of the measurement space that conserves as much
% of 'z' (in variance) as possible. This is not necessarily good for
% classification.

v = pca(prdat_without_labels, 0); figure; clf; plot(v);
%% as
[coeff,score,latent,tsquared,explained] = pca(getdata(prdat_without_labels));
explained % First 3 components explain 99,83% of all variability
scatter3(score(:,1),score(:,2),score(:,3))
axis equal
xlabel('1st Principal Component')
ylabel('2nd Principal Component')
zlabel('3rd Principal Component')

% K-means clsutering. Can be compared to the original data to "asses" the
% performance of the results.
kmeans_labels = prkmeans(prdat_without_labels, 2);
kmeans_dataset = prdataset(prdat_without_labels, kmeans_labels);
figure; clf; scatterdui(kmeans_dataset, 'legend');
% 
% % EM algorithm for mixture of Gaussians estimation
% prdat_soft = prdat_without_labels;
% prdat_soft = setlabtype(prdat_without_labels, 'soft');  % Set probabilistic labels
% [EM_labels, EM_mapping] = emclust(prdat_soft, nmc, 2);
% %figure; clf; scatterdui(prdat_soft, 'legend');
% %plotm(EM_mapping, [], 0.2:0.2:1);


%% PCA Training
PCA_mapping = prdat_without_labels*pcam(3);

true_error_qdc_pca = 0;
true_error_ldc_pca = 0;
true_error_nmsc_pca = 0;
true_error_knnc_pca = 0;

n = 100;
for i = 1 : n
    [prtrain_pca, prtest_pca]=gendat(prdat_without_labels*PCA_mapping, 0.65);
    W1_PCA = qdc(prtrain_pca);
    W2_PCA = ldc(prtrain_pca);
    W3_PCA = nmsc(prtrain_pca);
    W4_PCA = knnc(prtrain_pca);
    true_error_qdc_pca = true_error_qdc_pca + testc(prtest_pca*W1_PCA);  
    true_error_ldc_pca = true_error_ldc_pca + testc(prtest_pca*W2_PCA);
    true_error_nmsc_pca = true_error_nmsc_pca + testc(prtest_pca*W3_PCA);
    true_error_knnc_pca = true_error_knnc_pca + testc(prtest_pca*W4_PCA);
    if i == n
        true_error_qdc_pca = true_error_qdc_pca/n;
        true_error_ldc_pca = true_error_ldc_pca/n;
        true_error_nmsc_pca = true_error_nmsc_pca/n;
        true_error_knnc_pca = true_error_knnc_pca/n;
        disp('PCA: Gendat Error 100 repitions -- QDC -- LDC -- NMSC -- KNNC ')
        disp([true_error_qdc_pca true_error_ldc_pca true_error_nmsc_pca true_error_knnc_pca])
    end
end