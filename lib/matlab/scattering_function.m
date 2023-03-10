function [datas, scatter] = scattering_function(sub_color, images, labels, invScale, qfact1,qfact2, num_rotations)
    
    invariance_scale = invScale;
    quality_factors = [qfact1 qfact2];
    num_rotations = [num_rotations num_rotations];

    sn = waveletScattering2('ImageSize',size(images{1}, [1, 2]),'InvarianceScale',invariance_scale,'QualityFactors',quality_factors,'NumRotations',num_rotations);
    
    
    datafeatures = cell(length(images), 1);
    parfor i = 1:length(images)
        
        
        % Keep all scatter invariants
        %smat = featureMatrix(sn, images{i});
        % Mean only 3rd dimension
        %features = mean(smat, 3);
        
        %features = smat

        % Keep only highest order scatter
        smat = scatteringTransform(sn, images{i});
        % Reshape as scatter, size, size
        features = smat{size(smat, 1)}.images;
        features = reshape([features{:}], [], size(features{1}, 1), size(features{1}, 2));
        % Mean invariants for every scatter
        disp(size(features))
        %features = mean(features, 2:3)
        
        features = reshape(features, 1, [])
        datafeatures{i} = features;       
    end
    
    datas = {cell2mat(datafeatures),labels};

    save(sprintf(replace(fullfile("Data", sub_color, "scatter.mat"), '\', '/')), "datas")
    disp('done')

    scatter = sn;
end

