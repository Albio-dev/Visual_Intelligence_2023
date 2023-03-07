function [datas, scatter] = scattering_function(sub_color, images, labels, invScale, qfact1,qfact2, num_rotations)
    
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
%     images = images;
%     labels = labels;
       
    
%     invariance_scale = 32;
%     quality_factors = [3 1];
%     num_rotations = [4 4];
    disp(qfact1);
    disp(invScale);
    invariance_scale = invScale;
    quality_factors = [qfact1 qfact2];
    num_rotations = [num_rotations num_rotations];
    

    sn = waveletScattering2('ImageSize',size(images{1}, [1, 2]),'InvarianceScale',invariance_scale,'QualityFactors',quality_factors,'NumRotations',num_rotations);
    [~,npaths] = paths(sn);
    sum(npaths)
    coefficientSize(sn)
    
    
    datafeatures = cell(length(images), 1);
    parfor i = 1:length(images)
       
        smat = featureMatrix(sn, images{i});
        %features = mean(smat, 2:4);
        features = smat
        features = reshape(features, 1, []);
        
        datafeatures{i} = features;       
    end
    
    datas = {cell2mat(datafeatures),labels};

    save(sprintf(replace(fullfile("Data", sub_color, "scatter.mat"), '\', '/')), "datas")
    disp('done')

    scatter = sn;
end

