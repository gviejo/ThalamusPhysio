dset = 'Mouse12/Mouse12-120815';

path_to_data = '/mnt/DataGuillaume/MergedData/';

data_dir = fullfile(path_to_data,dset);
cd(data_dir);

%Parameters
binSize = 0.005; %in seconds
%binSize = 0.020;
%binSize = 0.200; % for decoding


%when the animal was exploring the arena
load('Analysis/BehavEpochs.mat','wakeEp');

%Spike Data
load('Analysis/SpikeData.mat', 'S', 'shank');
%Q = MakeQfromS(S,binSize);


%load position
[~,fbasename,~] = fileparts(pwd);
[X,Y,~,wstruct] = LoadPosition_Wrapper(fbasename);

%Load head-direction (wstuct is the raw position data, saves some time not
%to relaod the text file)
[ang,angGoodEp] = HeadDirection_Wrapper(fbasename,wstruct);

%and speed
%linSpd = LoadSpeed_Wrapper(fbasename,wstruct);

%Restrict exploration to times were the head-direction was correctly
%detected (you need to detect the blue and red leds, sometimes one of  the
%two is just not visible)
wakeEp  = intersect(wakeEp,angGoodEp);

%Restrict all data to wake (i.e. exploration)
%ang     = Restrict(ang,wakeEp);
X       = Restrict(X,wakeEp);
Y       = Restrict(Y,wakeEp);
% linSpd  = Restrict(linSpd,wakeEp);
% 
% %Note to regress spike Data to position, you need to get the same timestamps for the two measures. Easy:
% Xq = Restrict(X,Q);
% Yq = Restrict(Y,Q);
% Aq = Restrict(ang,Q);
% Sp = Restrict(linSpd,Q);
% cd(data_dir);    
% 
% dset = split(dset, '/');
% 
% 
% position = [Range(Q) Data(Xq) Data(Yq) Data(Aq) Data(Sp)];

%position = [Range(ang) Data(ang)];

position = [Range(X) Data(X) Data(Y)];



dlmwrite([fbasename '_XY.csv'], position, 'delimiter', ',', 'precision', 8);


    

%data_to_save = struct('X',  Data(Xq), 'Y',  Data(Yq), 'Ang', Data(Aq), 'speed', Data(Sp), 'ADn', dQadn(:,prefAngThIx), 'Pos', dQpos(:,prefAngPoIx));
%save('data_test_boosted_tree_05ms.mat', '-struct', 'data_to_save');

