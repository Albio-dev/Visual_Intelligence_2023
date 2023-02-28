sub_color = 'gray';
imagedir = fullfile('Data', sub_color);

Imds = imageDatastore(imagedir,'IncludeSubFolders',true,'FileExtensions',...
'.jpg','LabelSource','foldernames');

summary(Imds.Labels)

%%

images = readall(Imds);
labels = Imds.Labels;

%%

invariance_scale = 40;
quality_factors = 3;
num_rotations = 3;

sn = waveletScattering2('ImageSize',size(images{1}, [1, 2]),'InvarianceScale',invariance_scale,'QualityFactors',quality_factors,'NumRotations',num_rotations);
[~,npaths] = paths(sn);
sum(npaths)
coefficientSize(sn)

%%

datafeatures = [];
for i = 1:length(images)
   
    smat = featureMatrix(sn, images{i});
    features = mean(smat, 2:4);
    features = reshape(features, 1, []);
    
    datafeatures = [datafeatures; features];
end

%%


save(sprintf(replace(fullfile("Data", sub_color, "scatter.mat"), '\', '/')), "datafeatures")