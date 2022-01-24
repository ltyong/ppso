function [bestever,fitness,pop,fbest_hist]=ppso(layers,d,lb,ub,MAX_FES,funcid)

layersno = length(layers);
%生成初始粒子

layercumsum = cumsum(layers);
sz = sum(layers);
FES=0;
fitness = 1e200;


if d==50
    phi=0.04; 
elseif d==30
    phi=0.02;
else 
    phi=0.008;
end;
lvlp = zeros(layersno,max(layers),d);
lvlv = zeros(layersno,max(layers),d);
fitnesslayer = zeros(layersno,max(layers),1);
pblayer = zeros(layersno,max(layers),d);


XRRmin = repmat(lb, sz, 1);
XRRmax = repmat(ub, sz, 1);

p = XRRmin + (XRRmax - XRRmin) .* rand(sz, d);
v = 0.1*(XRRmin + (XRRmax - XRRmin) .* rand(sz, d));
pb = p;
gbf=1e200;

af=[];
if funcid==1
    af = Griewank(p);
end

FES = sz;
bestever = gbf;
fitness = af;
MAXGEN = floor(MAX_FES/sz);
fbest_hist=zeros(1,MAXGEN);

gen=1;
tp=zeros(sz,d);
tv=zeros(sz,d);
tpb=zeros(sz,d);
tf=9999999*zeros(sz,1);
layeridxs=zeros(layersno,2);
allpos=zeros(sz,d);
allv=zeros(sz,d);
while(FES < MAX_FES)
   
    [fv,pid]=sort(fitness);

    fbest_hist(1,gen)=fv(1);
    %金字塔分层-第1层
    idx = pid(1:layercumsum(1));
    lvlp(1,1:length(idx),:)=p(idx,:);
    lvlv(1,1:length(idx),:)=v(idx,:);
    fitnesslayer(1,1:length(idx),:)=fitness(idx);
    pblayer(1,1:length(idx),:)=pb(idx,:);
    
    %第2层以后
    for li=2:layersno
        idx = pid(layercumsum(li-1)+1:layercumsum(li));
        len = length(idx);
        lvlp(li,1:len,:)=p(idx,:);
        lvlv(li,1:len,:)=v(idx,:);
        fitnesslayer(li,1:len,:)=fitness(idx);
        pblayer(li,1:length(idx),:)=pb(idx,:);
    end
    
    ttidx=1;
    %除了第一层，其余每层的竞争和社会学习
    for li=layersno:-1:1
        lvlsize = layers(li);
        rlist = randperm(lvlsize);
        seprator = floor(lvlsize/2);
        rpairs = [rlist(1:seprator); rlist((seprator+1):(2*seprator))]';
        %competitive learning
        mask = (fitnesslayer(li,rpairs(:,1),1) > fitnesslayer(li,rpairs(:,2),1))';
        losers = mask.*rpairs(:,1) + ~mask.*rpairs(:,2);
        winners = ~mask.*rpairs(:,1) + mask.*rpairs(:,2);
        
        randco1 = rand(seprator, d);
        randco2 = rand(seprator, d);
        randco3 = rand(seprator, d);
        
        % losers learn from winners
        lvlvlosert=reshape(lvlv(li,losers,:),[seprator,d]);
        lvlplosert=reshape(lvlp(li,losers,:),[seprator,d]);
        lvlpblosert=reshape(pblayer(li,losers,:),[seprator,d]);
        
        lvlvwinert=reshape(lvlv(li,winners,:),[seprator,d]);
        lvlpwinert=reshape(lvlp(li,winners,:),[seprator,d]);
        lvlpbwinert=reshape(pblayer(li,winners,:),[seprator,d]);
        
        toplvlsize = layers(1);
        indciestop = 1+mod(randperm(seprator),ones(1,seprator)*toplvlsize);
        gbpmat = reshape(lvlp(1,indciestop,:),[seprator d]);
        
        lvlvlosert2 = randco1.*lvlvlosert;
        lvlvlosert2 = lvlvlosert2 + randco2.*(lvlpwinert - lvlplosert);
        lvlvlosert2 = lvlvlosert2 + randco3.*(lvlpblosert - lvlplosert);
        lvlplosert2 = lvlplosert + lvlvlosert2;
        
        
        if li~=1
            upperlvlsize = layers(li-1);
            indcies = 1+mod(randperm(seprator),ones(1,seprator)*upperlvlsize);
            upperpmat = reshape(lvlp(li-1,indcies,:),[seprator d]);

            randco1 = rand(seprator, d);
            randco2 = rand(seprator, d);
            randco3 = rand(seprator, d);
            randco4 = rand(seprator, d);


            lvlvwinert2 = randco1.*lvlvwinert;
            lvlvwinert2 = lvlvwinert2 + randco2.*(upperpmat - lvlpwinert);
            lvlvwinert2 = lvlvwinert2 + randco3.*(lvlpbwinert - lvlpwinert);
            lvlvwinert2 = lvlvwinert2 + phi*randco4.*(gbpmat- lvlpwinert);

            lvlvwinert2=reshape(lvlvwinert2,[seprator,d]);
            lvlpwinert2 = lvlpwinert + lvlvwinert2;
        else
            lvlvwinert2 = lvlvwinert;
            lvlpwinert2 = lvlpwinert;
        end
       
        mergedlvlp=[lvlplosert2;lvlpwinert2];
        mergelen = size(lvlplosert2,1);
        
        ts1=mergelen; ts2=size(lvlpwinert2,1);        
        layeridxs(li,:)=[ts1,ts2];
        
        mergedlvlv = [lvlvlosert2; lvlvwinert2];
        allpos(ttidx:ttidx+(ts1+ts2-1),:)=mergedlvlp;
        allv(ttidx:ttidx+(ts1+ts2-1),:)=mergedlvlv;
        ttidx = ttidx+(ts1+ts2);
        llosers{li}=losers;        
        wwiners{li}=winners;
    end
   
    allpos(allpos>ub)=ub;
    allpos(allpos<lb)=lb;
    if funcid==1
        ffs = Griewank(allpos);
    end
    ttidx=1;
    for li=layersno:-1:1    
        ts1=layeridxs(li,1);
        ts2=layeridxs(li,2);

        ff1 = ffs(ttidx:ttidx+ts1-1,:);
        lvlplosert2 = allpos(ttidx:ttidx+ts1-1,:);
        lvlvlosert2 = allv(ttidx:ttidx+ts1-1,:);
        ff2 = ffs(ttidx+ts1:ttidx+ts1+ts2-1,:);
        lvlpwinert2 = allpos(ttidx+ts1:ttidx+ts1+ts2-1,:);
        lvlvwinert2 = allv(ttidx+ts1:ttidx+ts1+ts2-1,:);
        ttidx=ttidx+ts1+ts2;
        
        losers= llosers{li};        
        winners = wwiners{li};
        
        
        goodloseridx = fitnesslayer(li,losers,1)'>ff1;
        goodwinderidx =fitnesslayer(li,winners,1)'>ff2;
        lidxs = losers(goodloseridx);
        widxs = winners(goodwinderidx);
        
        pblayer(li,lidxs,:)=lvlplosert2(goodloseridx,:);
        pblayer(li,widxs,:)=lvlpwinert2(goodwinderidx,:);        
       
        lvlv(li,losers,:) = lvlvlosert2;
        lvlp(li,losers,:) = lvlplosert2;
        
        lvlv(li,winners,:) = lvlvwinert2;
        lvlp(li,winners,:) = lvlpwinert2;
        
        fitnesslayer(li,losers,:) = ff1;
        fitnesslayer(li,winners,:) = ff2;
    end
    
    %%%resemble v and p
    for li=1:layersno
        if li==1
            leftb=1;
            rightb=layercumsum(li);
        else
            leftb=layercumsum(li-1)+1;
            rightb = layercumsum(li);
        end
        
        tp(leftb:rightb,:) = reshape(lvlp(li,1:layers(li),:),[layers(li) d]);
        tv(leftb:rightb,:) = reshape(lvlv(li,1:layers(li),:),[layers(li) d]);
        tpb(leftb:rightb,:) = reshape(pblayer(li,1:layers(li),:),[layers(li) d]);
        tf(leftb:rightb,:)=reshape(fitnesslayer(li,1:layers(li)),[layers(li) 1]);
    end
    
    p=tp;
    v=tv;
    pb=tpb;

    fitness = tf;
    bestever = min(bestever, min(fitness));
    FES = FES + sz;  
    gen=gen+1;
end
pop=p;

end