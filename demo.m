clc;
clear;
close all;
rng(74);
rt=10;
layers = [4 8 20 32];
ps=sum(layers);
d = 30;
MAX_FES = d*10000;
funcid = 1;
lb = -600;
ub = 600;
results= ones(1,rt)*99999999999;
for ri =1:rt
    [results(ri),fitness,pop,fbest]=ppso(layers,d,lb,ub,MAX_FES,funcid);  
    fprintf('%d : %e\n', ri,results(ri));
end
fprintf('\n\n====================\n\n');
fprintf('FID:%d mean reslut: %e\n', funcid,mean(results));

