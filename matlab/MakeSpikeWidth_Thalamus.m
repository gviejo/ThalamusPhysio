function MakeSpikeWidth_Thalamus()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% this program discriminate interneurons and pyramidal cells 
% It saves two values for each cell: half peak width and peak to peak
% both of them in ms
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[dumy fbasename dumy] = fileparts(pwd);

[spkWidth pk2pk meanWaveF maxIx] = Make_MeanWaveF(fbasename);

         
info = {'Peak 2 Peak';'spike Width (inverse of peak spectrum)';'mean wave forms';'maximal spk channel'};
SaveAnalysis(pwd,'SpikeWaveF',{pk2pk; spkWidth;meanWaveF;maxIx},{'pk2pk'; 'spkWidth'; 'meanWaveF';'maxIx'},info);
