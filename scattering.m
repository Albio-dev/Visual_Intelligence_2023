%% rgb
sub_color = "rgb";
datas = scattering_function(sub_color);
save(sprintf(replace(fullfile("Data", sub_color, "scatter.mat"), '\', '/')), "datas")
disp('done')
%% gray scale
sub_color = "gray";
datas = scattering_function(sub_color);
save(sprintf(replace(fullfile("Data", sub_color, "scatter.mat"), '\', '/')), "datas")
disp('done')

