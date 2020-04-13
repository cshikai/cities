load '../data.allprojectdata.mat
 
% merchant_district.csv as merchantdistrict matrix and column vectors both
% load akbank_custs_7sep
% load transactions trx4sep
% load carowners and districts36 : only 36 districts are considered
% load gdp.csv as matrix
% load time, time_original, attractiveness as tables

%% at first step we aim to find: 
% inflow, outflow, shopflow, workflow, frequency based flow,


customers = unique(cust_id); %cust_id from akbank trx cleaned
merchants = unique(mer_id); %mer_id from akbank trx cleaned
% districts = districts36; %districts list of 36 considered
freqdist = zeros(length(customers), length(districts)); %visit freq dist table

% filling customer WH districts demographic data in and if they have cars
% customer data = [id homedist workdist carowner];
customersdata = [customers zeros(length(customers),3)];
for i = 1:length(customers)
    for j = 1:length(CUSTID2)
        if customers(i) == CUSTID2(j)
            customersdata(i,2) = HDISTID(j);
            customersdata(i,3) = WDISTID(j);
        end
    end
    if ismember(customers(i) , carowners) % if they have cars
        customersdata(i,4) = 1;
    end
end


%filling the frequency table 
%each row is number of transactions of a customer in each district
for i = 1:length(cust_id) %iterate over transactions
    if rem(i,100000) == 0
        i
    end
    m = 0; n = 0;
    for j = 1:length(customers)
        if cust_id(i) == customers(j)
            m = j; %row number
            continue
        end
    end
    for k = 1:size(merchantdistrict,1)
        if mer_id(i) == merchantdistrict(k,1)
            n1 = merchantdistrict(k,2); %merchant of trx district
        end
    end
    for q = 1:length(districts)
        if districts(q) == n1
            n = q; %column number
            continue
        end
    end
    if (n~=0) && (m~=0)
        freqdist(m,n) = freqdist(m,n)+1; %updating frequency table
    end
end;

% whidfreqdist [dist cusotmer id, h, w, carowner, freq distriution]
whidfreqdist = [0 0 0 0 districts'; customersdata freqdist];

%aggregting home district levl frequencies
freqdist2 = [customersdata(:,2) freqdist];
aggfreq = [];
for i = 1:length(districts)
    i
    x = zeros(1,length(districts)+1);%id and districts
    x(1,1) = districts(i); %district id
    for j = 1:length(freqdist2)
        if districts(i) == freqdist2(j,1)
            x(1,2:length(districts)+1) = x(1,2:length(districts)+1)+freqdist2(j,2:length(districts)+1);
        end
    end
    aggfreq = [aggfreq; x];
end
    % aggfreq is [dist id, transactions to other districts]****
 aggfrequencies = aggfreq(:,2:size(aggfreq,2));


% %calculating fractions without considering home district
aggfreqother = aggfreq(:, 2:size(aggfreq,2));
% for i = 1:size(aggfreqother,2)
%     aggfreqother(i,i) = 0;
% end

% fraction distribution table ***including own district
aggtrxsum = zeros(size(aggfreqother,1),1);
for i = 1: size(aggfreqother,1)
    aggtrxsum(i) = sum(aggfreqother(i,:));
end

% aggfractions is the fractions of visit distribution***
aggfractions = zeros(size(aggfreqother));
for i = 1:size(aggfreqother,1)
    for j = 1:size(aggfreqother,2)
        aggfractions(i,j) = aggfreqother(i,j)/aggtrxsum(i);
    end
end
districtaggfractions = [0 districts'; districts aggfractions];


%% in-out flow calculations

workflow = zeros(length(districts),length(districts));
shopflow = zeros(length(districts),length(districts));
shopcarflow = zeros(length(districts),length(districts));
shoptransitflow = zeros(length(districts),length(districts));
shopfreqflow = zeros(length(districts),length(districts));
shopfreqflowcar = zeros(length(districts),length(districts));
shopuniqmerflow = zeros(length(districts),length(districts));
shopuniqmerflowcar = zeros(length(districts),length(districts));


for i= 1:length(districts)
    i %district id index
    
    
    % get customers with work in district i and not home in there
    workers = [];
    for j = 1:length(WDISTID)
        if (WDISTID(j) == districts(i)) && (HDISTID(j) ~= districts(i))
            workers = [workers; CUSTID2(j)];
        end
    end
    uniqworkers = unique(workers); % unique workers coming to i
    
    %finding workers' home
    workerhomedistid = zeros(length(uniqworkers));
    for j = 1:length(uniqworkers)
        for k = 1:length(CUSTID2)
            if uniqworkers(j) == CUSTID2(k)
                workerhomedistid(j) = HDISTID(k);
            end
        end
    end
        
    %finding # of workers from each origin
    for j = 1:length(districts)
        for k = 1:length(workerhomedistid)
            if (workerhomedistid(k) ==  districts(j)) && (workerhomedistid(k) ~=  districts(i))
                workflow(i,j) = workflow(i,j)+1; % coming to i from j for work
            end
        end
    end
    
   %---------------------------------------    
%     onlycusts = [];
%     for j = 1:length(uniqcusts)
%         g = 0;
%         for k = 1:length(uniqworkers)
%             if uniqcusts(j) == uniqworkers(k)
%                 g = g+1;
%             end
%         end
%         if g == 0
%             onlycusts = [onlycusts; uniqcusts(j)];
%         end
%     end
    %---------------------------------------

    %collectiong merchants in a particular district
    merchs = []; uniqmerchs = [];
    for j = 1:length(merch_dist_id)
        if districts(i) == merch_dist_id(j) 
            merchs = [merchs; merch_id(j)];
        end
    end
    uniqmerchs = unique(merchs); %unique merchants in district i
    
    %collecting merchants' customers
    custs = []; custmerchs = [];
    for j = 1:length(uniqmerchs)
        for k = 1:length(TMERID) %TMERID is merchant ids in trx table
            if TMERID(k) == uniqmerchs(j)
                custs = [custs; TCUSTID(k)];
                custmerchs = [custmerchs; TCUSTID(k) TMERID(k)]; %*******
                %cust visiting a merchant
            end
        end
    end
    transactions = length(custs); % # of transactions
    uniqcusts = unique(custs); % set of unique customers
    
    %finding shoppers' home who shop in i
    custdistid = zeros(length(uniqcusts),1);
    for j = 1:length(uniqcusts)
        for k = 1:length(CUSTID2)
            if uniqcusts(j) == CUSTID2(k)
                custdistid(j) = HDISTID(k);
                continue
            end
        end
    end
    custhome = [uniqcusts custdistid];
        
    
    
    %finding total # of customers from each origin
    for j = 1:length(districts)
        for k = 1:length(custdistid)
            if (custdistid(k) ==  districts(j)) && (custdistid(k) ~=  districts(i))
                shopflow(i,j) = shopflow(i,j)+1;
            end
        end
    end
    %finding # of customers from each origin wcar or w/ocar
    for j = 1:length(districts)
        for k = 1:length(custdistid)
            if (custdistid(k) ==  districts(j)) && (custdistid(k) ~=  districts(i)) && (ismember(custdistid(k),carowners))
                shopcarflow(i,j) = shopcarflow(i,j)+1;
            end
        end
    end
    for j = 1:length(districts)
        shoptransitflow(i,j) = shopflow(i,j) - shopcarflow(i,j);
    end
    
    % customer shopping frequency
    % number of unique merchant visits by a customer
    custuniqmer = [];
    custfreqany = [];
    for j = 1:length(uniqcusts)
        visitedmerchs = [];
        freqany = 0;
        for k = 1:size(custmerchs,1)
            if uniqcusts(j) == custmerchs(k,1)
                visitedmerchs = [visitedmerchs; custmerchs(k,2)];
                freqany = freqany + 1;
            end
        end
        custfreqany = [custfreqany; uniqcusts(j) custdistid(j) freqany ismember(uniqcusts(j),carowners)]; % all visits
        % unique merchant visits
        custuniqmer = [custuniqmer; uniqcusts(j) custdistid(j) length(unique(visitedmerchs)) ismember(uniqcusts(j),carowners)];
    end
    
    %finding total # and unique merchant of visits by customers from each origin
    
    for j = 1:length(districts)
        for k = 1:size(custfreqany,1)
            if (custfreqany(k,2) == districts(j)) && (custfreqany(k,2) ~=  districts(i))
                shopfreqflow(i,j) = shopfreqflow(i,j)+custfreqany(k,3);% all shoppers frequency
                % car owners frequency
                shopfreqflowcar(i,j) = shopfreqflowcar(i,j)+custfreqany(k,3)*custfreqany(k,4);
                % all shoppers unique merchant visits
                shopuniqmerflow(i,j) = shopuniqmerflow(i,j)+custuniqmer(k,3);
                % all shoppers unique merchant visits car owners
                shopuniqmerflowcar(i,j) = shopuniqmerflowcar(i,j)+custuniqmer(k,3)*custuniqmer(k,4);
            end
        end
    end
    
end


% work in/out flow
inworkflow = []; outworkflow =[];
for i = 1:size(workflow,1)
    inworkflow = [inworkflow; sum(workflow(i,:))];
    outworkflow = [outworkflow; sum(workflow(:,i))];
end


% shop in/out flow
inshopflow = []; outshopflow =[];
for i = 1:size(shopflow,1)
    inshopflow = [inshopflow; sum(shopflow(i,:))];
    outshopflow = [outshopflow; sum(shopflow(:,i))];
end


% shop frequency flow
inshopfreqflow = []; outshopfreqflow =[];
for i = 1:size(shopfreqflow,1)
    inshopfreqflow = [inshopfreqflow; sum(shopfreqflow(i,:))];
    outshopfreqflow = [outshopfreqflow; sum(shopfreqflow(:,i))];
end

% shop unique visit flow
inshopmerflow = []; outshopmerflow =[];
for i = 1:size(shopuniqmerflow,1)
    inshopmerflow = [inshopmerflow; sum(shopuniqmerflow(i,:))];
    outshopmerflow = [outshopmerflow; sum(shopuniqmerflow(:,i))];
end


% total in/out flow
inflow = inshopflow+inworkflow;
outflow = outshopflow + outworkflow;

% full flow results matrix
datamat = [districts inflow outflow inworkflow outworkflow inshopflow outshopflow ...
    inshopfreqflow outshopfreqflow inshopmerflow outshopmerflow];
% xlswrite('flowmatrix.xlsx',datamat);

% flow by individuals
x0 = ones(size(gdp,1),1); 
x1 = datamat(:,2); x2 = datamat(:,3); % in/out total flow
X1 = [x0 x1 x2];

% flow by work/shop
x3 = datamat(:,4); x4 = datamat(:,5); x5 = datamat(:,6); x6 = datamat(:,7); %work/shop in/out
X2 = [x0 x3 x4 x5 x6];

% flow by frequency
x7 = datamat(:,8); x8 = datamat(:,9);
X3 = [x0 x4 x5 x7 x8];
X4 = [x0 x4+x7 x5+x8];

% flow by unique visits
x9 = datamat(:,10); x10 = datamat(:,11);
X5 = [x0 x4 x5 x9 x10];
X6 = [x0 x4+x9 x5+x10];

y = gdp(:,3); %gdp2015

% controling for population effect
X10 = [x0 x1 x2 population];

% plotmatrix(X!)
Z = [X3 X4 X5 X6];
plotmatrix(Z,y)
title('shop-in-degree, shop-out-degree, work-in-degree, work-out-degree, VS GDP')

[b,bint,r,rint,stats] = regress(y,X10);
scatter(r,y)
lsline
corr(r,y)
dlmwrite('notnormalized_workshopinoutdegree.csv', Z,'precision',20);

%%

winin = zeros(36,1);
for i = 1:36
    for j = 1:length(WDISTID)
        if (WDISTID(j) == districts(i))&&(HDISTID(j) == districts(i))
            winin(i) = winin(i)+1;
        end
    end
end



workflow2 = zeros(length(districts),length(districts));
shopflow2 = zeros(length(districts),length(districts));
shopcarflow2 = zeros(length(districts),length(districts));
shoptransitflow2 = zeros(length(districts),length(districts));
shopfreqflow2 = zeros(length(districts),length(districts));
shopfreqflowcar2 = zeros(length(districts),length(districts));
shopuniqmerflow2 = zeros(length(districts),length(districts));
shopuniqmerflowcar2 = zeros(length(districts),length(districts));



for i= 1:length(districts)
    i %district id index
    
    
    % get customers with work in district i and not home in there
    workers2 = [];
    for j = 1:length(WDISTID)
        if (WDISTID(j) == districts(i)) && (HDISTID(j) == districts(i))
            workers2 = [workers2; CUSTID2(j)];
        end
    end
    uniqworkers2 = unique(workers2); % unique workers coming to i
    
    %finding workers' home
    workerhomedistid2 = zeros(length(uniqworkers2));
    for j = 1:length(uniqworkers2)
        for k = 1:length(CUSTID2)
            if uniqworkers2(j) == CUSTID2(k)
                workerhomedistid2(j) = HDISTID(k);
            end
        end
    end
        
    %finding # of workers from each origin
    for j = 1:length(districts)
        for k = 1:length(workerhomedistid2)
            if (workerhomedistid2(k) ==  districts(j)) && (workerhomedistid2(k) ==  districts(i))
                workflow2(i,j) = workflow2(i,j)+1; % coming to i from j for work
            end
        end
    end


    %collectiong merchants in a particular district
    merchs2 = []; uniqmerchs2 = [];
    for j = 1:length(merch_dist_id)
        if districts(i) == merch_dist_id(j) 
            merchs2 = [merchs2; merch_id(j)];
        end
    end
    uniqmerchs2 = unique(merchs2); %unique merchants in district i
    
    %collecting merchants' customers
    custs2 = []; custmerchs2 = [];
    for j = 1:length(uniqmerchs2)
        for k = 1:length(TMERID) %TMERID is merchant ids in trx table
            if TMERID(k) == uniqmerchs2(j)
                custs2 = [custs2; TCUSTID(k)];
                custmerchs2 = [custmerchs2; TCUSTID(k) TMERID(k)]; %*******
                %cust visiting a merchant
            end
        end
    end
    transactions2 = length(custs2); % # of transactions
    uniqcusts2 = unique(custs2); % set of unique customers
    
    %finding shoppers' home who shop in i
    custdistid2 = zeros(length(uniqcusts2),1);
    for j = 1:length(uniqcusts2)
        for k = 1:length(CUSTID2)
            if uniqcusts2(j) == CUSTID2(k)
                custdistid2(j) = HDISTID(k);
                continue
            end
        end
    end
    custhome2 = [uniqcusts2 custdistid2];
        
    
    
    %finding total # of customers from each origin
    for j = 1:length(districts)
        for k = 1:length(custdistid2)
            if (custdistid2(k) ==  districts(j)) && (custdistid2(k) ==  districts(i))
                shopflow2(i,j) = shopflow2(i,j)+1;
            end
        end
    end
    %finding # of customers from each origin wcar or w/ocar
    for j = 1:length(districts)
        for k = 1:length(custdistid2)
            if (custdistid2(k) ==  districts(j)) && (custdistid2(k) ==  districts(i)) && (ismember(custdistid2(k),carowners))
                shopcarflow2(i,j) = shopcarflow2(i,j)+1;
            end
        end
    end
    for j = 1:length(districts)
        shoptransitflow2(i,j) = shopflow2(i,j) - shopcarflow2(i,j);
    end
    
    % customer shopping frequency
    % number of unique merchant visits by a customer
    custuniqmer2 = [];
    custfreqany2 = [];
    for j = 1:length(uniqcusts2)
        visitedmerchs2 = [];
        freqany2 = 0;
        for k = 1:size(custmerchs2,1)
            if uniqcusts2(j) == custmerchs2(k,1)
                visitedmerchs2 = [visitedmerchs2; custmerchs2(k,2)];
                freqany2 = freqany2 + 1;
            end
        end
        custfreqany2 = [custfreqany2; uniqcusts2(j) custdistid2(j) freqany2 ismember(uniqcusts2(j),carowners)]; % all visits
        % unique merchant visits
        custuniqmer2 = [custuniqmer2; uniqcusts2(j) custdistid2(j) length(unique(visitedmerchs2)) ismember(uniqcusts2(j),carowners)];
    end
    
    %finding total # and unique merchant of visits by customers from each origin
    
    for j = 1:length(districts)
        for k = 1:size(custfreqany2,1)
            if (custfreqany2(k,2) == districts(j)) && (custfreqany2(k,2) ==  districts(i))
                shopfreqflow2(i,j) = shopfreqflow2(i,j)+custfreqany2(k,3);% all shoppers frequency
                % car owners frequency
                shopfreqflowcar2(i,j) = shopfreqflowcar2(i,j)+custfreqany2(k,3)*custfreqany2(k,4);
                % all shoppers unique merchant visits
                shopuniqmerflow2(i,j) = shopuniqmerflow2(i,j)+custuniqmer2(k,3);
                % all shoppers unique merchant visits car owners
                shopuniqmerflowcar2(i,j) = shopuniqmerflowcar2(i,j)+custuniqmer2(k,3)*custuniqmer2(k,4);
            end
        end
    end
    
end

% work in/out flow
inworkflow2 = []; outworkflow2 =[];
for i = 1:size(workflow2,1)
    inworkflow2 = [inworkflow2; sum(workflow2(i,:))];
    outworkflow2 = [outworkflow2; sum(workflow2(:,i))];
end


% shop in/out flow
inshopflow2 = []; outshopflow2 =[];
for i = 1:size(shopflow,1)
    inshopflow2 = [inshopflow2; sum(shopflow2(i,:))];
    outshopflow2 = [outshopflow2; sum(shopflow2(:,i))];
end


% shop frequency flow
inshopfreqflow2 = []; outshopfreqflow2 =[];
for i = 1:size(shopfreqflow2,1)
    inshopfreqflow2 = [inshopfreqflow2; sum(shopfreqflow2(i,:))];
    outshopfreqflow2 = [outshopfreqflow2; sum(shopfreqflow2(:,i))];
end

% shop unique visit flow
inshopmerflow2 = []; outshopmerflow2 =[];
for i = 1:size(shopuniqmerflow2,1)
    inshopmerflow2 = [inshopmerflow2; sum(shopuniqmerflow2(i,:))];
    outshopmerflow2 = [outshopmerflow2; sum(shopuniqmerflow2(:,i))];
end

% total in/out flow
inflow2 = inshopflow2+inworkflow2;
outflow2 = outshopflow2 + outworkflow2;