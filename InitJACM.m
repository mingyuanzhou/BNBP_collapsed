function [X,WO] = InitJACM

fid = fopen('jacm.dat');
C = textscan(fid, '%s');
C = C{1};
fclose(fid);
ii = -ones(ceil(length(C)/2),1);
jj = -ones(ceil(length(C)/2),1);
ss = -ones(ceil(length(C)/2),1);
count = 0;
docnum = 0;
for i=1:length(C)    
    temp = textscan(C{i},'%d','delimiter', ':');
    if length(temp{1})==1
        docnum = docnum+1;
    else
        count = count+1;        
        ii(count) = temp{1}(1)+1;
        jj(count) = docnum;
        ss(count) = temp{1}(2);
    end
end
ii(count+1:end)=[];
jj(count+1:end)=[];
ss(count+1:end)=[];
X = sparse(ii,jj,ss);
dex = (sum(X>0,2)<5);
X = X(~dex,:);

fid = fopen('jacm-voc.dat');
C = textscan(fid, '%s');
C = C{1};
fclose(fid);
WO = C;
WO = WO(~dex);
save JACM_MZ.mat WO X