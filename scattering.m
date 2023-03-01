sub_color = 'rgb';
imagedir = fullfile('Data', sub_color);

Imds = imageDatastore(imagedir,'IncludeSubFolders',true,'FileExtensions',...
'.jpg','LabelSource','foldernames');


summary(Imds.Labels)

%%

images = readall(Imds);
labels = Imds.Labels;

labels = labels(:) ~= 'dog';

%%

invariance_scale = 128;
quality_factors = [4 2];
num_rotations = [8 8];

sn = waveletScattering2('ImageSize',size(images{1}, [1, 2]),'InvarianceScale',invariance_scale,'QualityFactors',quality_factors,'NumRotations',num_rotations);
[~,npaths] = paths(sn);
sum(npaths)
coefficientSize(sn)

%%

datafeatures = cell(length(images), 1);
parfor i = 1:length(images)
   
    smat = featureMatrix(sn, images{i});
    features = mean(smat, 2:4);
    features = reshape(features, 1, []);
    
    datafeatures{i} = features;
end

datas = {cell2mat(datafeatures),labels};
%%


save(sprintf(replace(fullfile("Data", sub_color, "scatter.mat"), '\', '/')), "datas")
disp('done')
