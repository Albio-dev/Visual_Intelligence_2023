function datas = scattering_function(sub_color, data, invScale, qualityFactors, num_rotations)
    
%     imagedir = fullfile('Data', sub_color);
%     
%     Imds = imageDatastore(imagedir,'IncludeSubFolders',true,'FileExtensions',...
%     '.jpg','LabelSource','foldernames');
%     
%     
%     summary(Imds.Labels)
    
    
%     images = readall(Imds);
%     labels = Imds.Labels;
%     
%     labels = labels(:) ~= 'dog';
    images = data(0);
    labels = data(1);
       
    
%     invariance_scale = 32;
%     quality_factors = [3 1];
%     num_rotations = [4 4];
    invariance_scale = invScale;
    quality_factors = qualityFactors;
    

    sn = waveletScattering2('ImageSize',size(images{1}, [1, 2]),'InvarianceScale',invariance_scale,'QualityFactors',quality_factors,'NumRotations',num_rotations);
    [~,npaths] = paths(sn);
    sum(npaths)
    coefficientSize(sn)
    
    
    datafeatures = cell(length(images), 1);
    parfor i = 1:length(images)
       
        smat = featureMatrix(sn, images{i});
        features = mean(smat, 2:4);
        features = reshape(features, 1, []);
        
        datafeatures{i} = features;
    end
    
    datas = {cell2mat(datafeatures),labels};
end

