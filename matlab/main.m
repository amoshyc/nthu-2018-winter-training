clc;
clear;

img_paths = {...
    './data/violet.jpg'; ...
    './data/kim.jpg'; ...
    './data/osas.png'; ...
};

for i = 1:3
    subplot(2, 3, i + 3)
    rgbs{i} = imread(img_paths{i});
    lbps{i} = lbp(rgb2gray(rgbs{i}));
%     lbps{i} = lbp2(rgb2gray(rgbs{i}));
    lbps{i}.Normalization = 'probability';
end

T = lbps{1}.Values;
for i = 1:3
    I = lbps{i}.Values;
    similarity = dot(I, T) / norm(I) / norm(T);
    subplot(2, 3, i)
    imshow(rgbs{i});
    subplot(2, 3, i + 3)
    title(num2str(similarity, '%.3f'));
end
