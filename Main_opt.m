%function [] = test(problem_number,t0,t1)
clc; clear; close all
%% define some basic parameters
warning off
num_initial = 100;
max_evaluation = 1000;
problem_number=5;
num_vari=10; %dim
size_pop=5*num_vari;
size_a=2*num_vari;
his=[];
num_iteration=50; %runs

%% the main loop
for iter=1:num_iteration
clearvars -except bestval_iter iter rbest problem_number ts his t0 t1 size_pop size_a num_vari num_initial max_evaluation;% close all;
s=tic;
rng(iter,'philox')
%figure
%% select the problem
fun_name = @(x)problem_test(x,problem_number);

switch problem_number
    case 1,    bd=5.12; 
    case 2,    bd=2.048;  
    case 3,    bd=32.768; 
    case 4,    bd=600;    
    case 5,    bd=5;  
end
design_space=[ones(1,num_vari)*-bd;ones(1,num_vari)*bd];
XVmin=design_space(1,:); XVmax=design_space(2,:);
DEsetting; 
rbest=[]; sample_y_old=[]; renewtime=0;

%% initilize the population
sample_x = repmat(design_space(1,:),num_initial,1) + repmat(design_space(2,:)-design_space(1,:),num_initial,1).*lhsdesign(num_initial,num_vari,'criterion','maximin','iterations',1000);
sample_y = feval(fun_name,sample_x); evalution=size(sample_x,1); srgtSRGT = Trainrbf(sample_x,sample_y);
[~,b]=sort(sample_y); initial_pop=sample_x(b(1:size_pop),:);  initialpop=initial_pop;
initialpop=initial_pop; source=ones(size(sample_x,1),1); srgtSRGT = Trainrbf(sample_x,sample_y); i=0;
[~,A]=sort(sample_y); Ay=sample_y(A(1:size_a),:);  A=sample_x(A(1:size_a),:);

%%
while 1
    %% global evolution
    i=i+1;
    FUN=@(x)srgtsRBFEvaluate(x, srgtSRGT); 
    [pop,mu_F,mu_CR,sample_x,srgtSRGT] =  SADE(FUN, fun_name, {num_vari,setDE,setJA}, -inf, 1, initial_pop, A, XVmin, XVmax, sample_x, srgtSRGT, problem_number);
    initial_pop=pop; setJA.JA_init_mu_F=mu_F; F(i)=mu_F; CR(i)=mu_CR; setJA.JA_init_mu_CR=mu_CR; sample_y=feval(fun_name,sample_x); eval=size(sample_x,1); source=[source;2];
    record_std(i,:) = std([pop;A]);
    
    %% local search
    nsn=min((num_vari+1)*(num_vari+2)/2,floor(size(sample_x,1)/4));
    
    for j = 1: size(pop,1)
        bestmem=pop(j,:); dis=sum((bestmem-sample_x).^2,2).^.5; [~,b]=sort(dis); cx=sample_x(b(1:nsn),:); 
        Xmax=max(cx);  Xmin=min(cx); % Xmin=ones(1,num_vari)*-1; Xmax=ones(1,num_vari)*1; % 
        infill_criterion=@(x)(srgtsRBFEvaluate(x, srgtSRGT));
        options=optimset('Display','off','algorithm','interior-point','UseParallel','never','MaxFunEvals',300) ;
        if ~isnan(srgtsRBFEvaluate(bestmem, srgtSRGT)), [best_x, val , ~ , details]=fmincon(infill_criterion, bestmem ,[],[],[],[],Xmin, Xmax, [],options); %, dmodel
            un=srgtsRBFEvaluate_un(best_x, srgtSRGT);
            offx(j,:)=best_x; offy(j,:)=val;  check(j,1)=un;
            ds=best_x-sample_x; ds=sum(ds.^2,2).^0.5; ds=min(ds); checkds(j,1)=ds;
        end
    end
    
    [~,b]=sort(sample_y); bestmem=sample_x(b(1),:); dis=sum((bestmem-sample_x).^2,2).^.5; [~,b]=sort(dis); cx=sample_x(b(1:nsn),:);
    infill_criterion=@(x)(srgtsRBFEvaluate(x, srgtSRGT));
    options=optimset('Display','off','algorithm','interior-point','UseParallel','never','MaxFunEvals',300) ;
    if ~isnan(srgtsRBFEvaluate(bestmem, srgtSRGT)), [best_x, val , ~ , details]=fmincon(infill_criterion, bestmem ,[],[],[],[],XVmin, XVmax, [],options); %, dmodel
        offx(size(pop,1)+1,:)=best_x; offy(size(pop,1)+1,:)=val;
        un=srgtsRBFEvaluate_un(best_x, srgtSRGT); check(size(pop,1)+1,1)=un;
        ds=best_x-sample_x; ds=sum(ds.^2,2).^0.5; ds=min(ds); checkds(size(pop,1)+1,1)=ds; 
    end
    
    check=1-(checkds)./max(checkds)+rand(size(pop,1)+1,1)*0.1; offy=offy.*check;
    [~,b]=sort(offy);  best_x=offx(b(1),:); ds=best_x-sample_x; ds=sum(ds.^2,2).^0.5; ds=min(ds);
    source=[source;3];  sample_x=[sample_x;best_x]; sample_y=feval(fun_name,sample_x);
    
    
    srgtSRGT = Trainrbf(sample_x,sample_y);
    fprintf('iter %d  evaluation: %d, current best solution: %f     %d      %f\n',iter, size(sample_x,1), min(sample_y),b(1),ds); 
    bestval(i)=min(sample_y);  xx(i)=size(sample_x,1);
    %semilogy(xx,bestval); drawnow;

    %% update A
    c1=sample_x(end,:); dc1=c1-A; dc1=sum(dc1.^2,2).^0.5; dc1max=max(dc1); dc1min=min(dc1); pro1=dc1min/dc1max; 
    if rand(1)< pro1 & sample_y(end,:)<max(Ay), [~,b]=max(Ay);  A(b,:)=c1; Ay(b,:)=sample_y(end,:); end
    c2=sample_x(end-1,:);  dc2=c2-A; dc2=sum(dc2.^2,2).^0.5; dc2max=max(dc2); dc2min=min(dc2); pro2=dc2min/dc2max;
    if rand(1)< pro2 & sample_y(end-1,:)<max(Ay), [~,b]=max(Ay);  A(b,:)=c2; Ay(b,:)=sample_y(end-1,:); end
    
    %% stop?
    evaluation=size(sample_x,1); if evaluation>max_evaluation, max_evaluation=1000; break; end
    
    %if min(sample_y)<5e-2, break; end
    %% restart?
    if max(std(pop))<1e-2 & abs( min(sample_y(end-19:end,1)) - min(sample_y(1:end-20,1)) ) / min(sample_y(1:end-20,1) )<1e-3 %&& ds<0
        clc; fprintf('restart \n'); 
        renewtime=renewtime+1;
        initial_pop=initialpop;
        if renewtime==1, sample_y_fin=sample_y; 
        else, sample_y_fin=[ sample_y_fin; sample_y(101:end,:)]; end
        sample_x_old=sample_x(101:end,:); sample_y_old=sample_y(101:end,:);
        sample_x(101:end,:)=[]; sample_y(101:end,:)=[]; 
        max_evaluation=1000-size(sample_y_fin,1)+100;
    end
    %
end
%% record some history data
if renewtime>0, sample_y_fin=[sample_y_fin;sample_y(101:end,:)];
else sample_y_fin=sample_y; end
%sample_y_fin=[sample_y(1:1001)];
ts(iter)=toc(s);
bestval_iter(iter)=min(sample_y_fin);
his=[his,sample_y_fin(1:1001)];
end

%% plot
for j=1:size(his,1)
    hisbesty(j,1:iter)=min(his(1:j,:));
end
proposed=mean(hisbesty,2);
maker_idx=[1:50:1001];
semilogy([100:1000],proposed(100:1000),'-sk','MarkerIndices',maker_idx,'MarkerSize',6,'MarkerFaceColor','k','MarkerEdgeColor','k','LineWidth',1); hold on

%% save the results
save([num2str(num_vari),'D_f',num2str(problem_number),'_',num2str(iter),'runs']);
%end
