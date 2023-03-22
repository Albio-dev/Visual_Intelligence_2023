function [sn] = get_scatterNet(invariance_scale, quality_factors, num_rotations, images_size)

    sn = waveletScattering2('ImageSize',double(cell2mat(images_size)),          ...
                            'InvarianceScale',double(invariance_scale),         ...
                            'QualityFactors',double(cell2mat(quality_factors)), ...
                            'NumRotations',double(cell2mat(num_rotations)));    ...

end
    