%% rgb
sub_color = "rgb";
datas = scattering_function(sub_color,32,[3 1],[4 4]);
save(sprintf(replace(fullfile("Data", sub_color, "scatter.mat"), '\', '/')), "datas")
disp('done')
%% gray scale
sub_color = "gray";
datas = scattering_function(sub_color);
save(sprintf(replace(fullfile("Data", sub_color, "scatter.mat"), '\', '/')), "datas")
disp('done')

%%
ass = [1,3];

a = ass(1);