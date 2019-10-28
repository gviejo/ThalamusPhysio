path_to_data = '/mnt/DataGuillaume/MergedData';

file = List2Cell(fullfile(path_to_data,'datasets_ThalHpc.list')); 

for i = 1:size(file)
    dset = file(i)
    data_dir = fullfile(path_to_data, char(dset));
    cd(data_dir);
    [~,fbasename,~] = fileparts(pwd);
    
    %when the animal was exploring the arena
    load('Analysis/BehavEpochs.mat','wakeEp');

    %Spike Data
    load('Analysis/SpikeData.mat', 'S', 'shank');

    %All infor regarding HD cells
    load('Analysis/HDCells.mat'); 

    %needed to know on which electrode group each cell is recorded
    load('Analysis/GeneralInfo.mat', 'shankStructure'); 
    
    %load position
    [~,fbasename,~] = fileparts(pwd);
    [X,Y,~,wstruct] = LoadPosition_Wrapper(fbasename);
    [ang,angGoodEp] = HeadDirection_Wrapper(fbasename,wstruct);

    
    linSpd = LoadSpeed_Wrapper(fbasename,wstruct);

    speed = [Range(linSpd), Data(linSpd)];
    
    save(fullfile(data_dir, char('Analysis/linspeed.mat')), 'speed');
    
end
    