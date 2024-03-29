function [data] = scattering(images, sn)
   
    len = size(images, 1);

    data = single(images);   

    

    %smat = featureMatrix(sn, squeeze(data(1, :, :)));

    
    datafeatures = cell(len, 1);
    parfor i = 1:len

        
        
        % Keep all scatter invariants
        %smat = featureMatrix(sn, squeeze(data(i, :, :)));
        % Mean only 3rd dimension
        %features = mean(smat, 3);
        %features = smat
        

        % Keep only highest order scatter
        smat = scatteringTransform(sn, squeeze(data(i, :, :)));

        % Reshape as scatter, size, size
        features = smat{size(smat, 1)}.images;
        features = reshape([features{:}], [], size(features{1}, 1), size(features{1}, 2));
        
        features = reshape(features, 1, []);
        datafeatures{i} = features;
    end
    
    data = cell2mat(datafeatures);

    %save(sprintf(replace(fullfile("Data", sub_color, "scatter.mat"), '\', '/')), "datas")
    disp('done')

end

