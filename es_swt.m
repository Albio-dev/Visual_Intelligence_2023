%% Load the data
unzip(fullfile('physionet_ECG_data-master.zip'))
unzip(fullfile('physionet_ECG_data-master','ECGData.zip'),...
    fullfile('ECGData'))

load(fullfile('ECGData','ECGData.mat'))

%helperPlotRandomRecords(ECGData,14)

if ~exist('Data/', 'dir')
       mkdir('Data/')
end


%% Extract the data
ecg = ECGData.Data;
labels = string(ECGData.Labels);
%labels = data.Labels;

% NSR: normal subject
% ARR: cardiac arrhythmia subject
% CHF: heart failure subject
labels(find(labels == 'NSR')) = 0;
labels(find(labels == 'ARR')) = 1;
labels(find(labels == 'CHF')) = 2;
labels = str2double(labels);


%% Visualize the data
row = 3;
col = 1;
figure("Name",'Full lenght')
subplot(row,col,1); plot(0:65535,ecg(1,:)); title('ARR signal'); xlabel('Samples'); ylabel('Signal'); grid; axis('tight');
subplot(row,col,2); plot(0:65535,ecg(100,:)); title('CHF signal'); xlabel('Samples'); ylabel('Signal'); grid; axis('tight');
subplot(row,col,3); plot(0:65535,ecg(150,:)); title('NSR signal'); xlabel('Samples'); ylabel('Signal'); grid; axis('tight');

figure("Name",'Subset')
subplot(row,col,1); plot(0:2559,ecg(1,1:2560)); title('ARR signal'); xlabel('Samples'); ylabel('Signal'); grid; axis('tight');
subplot(row,col,2); plot(0:2559,ecg(100,1:2560)); title('CHF signal'); xlabel('Samples'); ylabel('Signal'); grid; axis('tight');
subplot(row,col,3); plot(0:2559,ecg(150,1:2560)); title('NSR signal'); xlabel('Samples'); ylabel('Signal'); grid; axis('tight');


%% Compute Stationary Wavelet Transform
num_levels = 6;
[num_subjs, num_features] = size(ecg);
window_size = 2560;    % 20 seconds (512s total)
num_windows = 25;     
features_multi_subjects = zeros(num_subjs*num_windows,num_levels+1);

j = 1;
for s=1:num_subjs
    % Get the ecg for subject: s
    temp_ecg = ecg(s,1:window_size*num_windows);
    for w=0:num_windows - 1
        % Extract the window of 20 seconds
        if w == 0
            window = temp_ecg(1:window_size*(w+1));
        else
            window = temp_ecg((window_size*w)+1:window_size*(w+1));
        end
        
        % Compute the SWT
        dwtmode('per');
        [swa,swd] = swt(window,num_levels,'db2');
        details_variance = var(swd,0,2);
        features_multi_subjects(j,1:num_levels) = details_variance;
        approx_variance_ll = var(swa(num_levels,:));
        features_multi_subjects(j,num_levels+1) = approx_variance_ll;

        j = j + 1;
    end
end

%%  Adjust the labels vector
new_labels = repelem(labels,num_windows);

%% Save the features
data.data = features_multi_subjects;
data.labels = new_labels;
save('Data/features.mat',"data")

%% Compute fft module for new data
features_multi_subjects_fourier = zeros(num_subjs*num_windows,num_levels+1);
for i=1:size(features_multi_subjects,1)
    features_multi_subjects_fourier(i,:) = abs(fft(features_multi_subjects(i,:)));
end

%% Save the features from Fourier transform
data_fourier.data = features_multi_subjects_fourier;
data_fourier.labels = new_labels;
save('Data/features_fourier.mat',"data")








