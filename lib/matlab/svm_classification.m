load Data/rgb/scatter.mat
%%
scattering = datas{1};
labels = cell2mat(datas{2});

idx = randperm(size(scattering,1),round(size(scattering,1)*0.8));
trainfeatures = [];
train_labels = [];
idx_test = 1:size(scattering,1);
for i = 1:length(idx)
    trainfeatures =  cat(1,trainfeatures,scattering(idx(i),:));
    train_labels = cat(1,train_labels,labels(idx(i)));
    disp(idx(i))
    idx_test(idx_test == idx(i)) = [];
end

testfeatures = [];
test_labels = [];
for ii = 1:length(idx_test)
    testfeatures = cat(1,testfeatures,scattering(idx_test(ii),:));
    test_labels = cat(1,test_labels,labels(idx_test(ii)));
end


%%
template = templateSVM(...
'KernelFunction', 'polynomial', ...
'PolynomialOrder', 3, ...
'KernelScale', 1, ...
'BoxConstraint', 314, ...
'Standardize', true);

classificationSVM = fitcecoc(trainfeatures,train_labels, 'Learners',template, 'Coding', 'onevsall');


%%

kfoldmodel = crossval(classificationSVM, 'KFold', 5);
loss = kfoldLoss(kfoldmodel)*100;
crossvalAccuracy = 100-loss

[predLabels,scores] = predict(classificationSVM,testfeatures);
testAccuracy = ...
sum(predLabels== test_labels)/numel(test_labels)*100

%%

figure
cchart = confusionchart(test_labels,predLabels);
cchart.Title ={'Confusion Chart for Wavelet';
'Scattering Features using SVM'};
cchart.RowSummary = 'row-normalized';
cchart.ColumnSummary = 'column-normalized';
