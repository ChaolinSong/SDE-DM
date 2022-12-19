function [pop_de,mu_F,mu_CR,sample_x,srgtSRGT] = SADE(FUN, fun_name, argumentPack, ftarget, maxfunevals, initial_pop, Ar, XVmin, XVmax, sample_x,srgtSRGT,problem_number)

    % extract second argument
    DIM = argumentPack{1};
    setDE = argumentPack{2};
    setJA = argumentPack{3};
        
    popsize = size(initial_pop,1);%min(eval(setDE.popsize), setDE.maxpopsize); %   % set parental population size e.g. 4 + floor(3 * log(DIM))
    offspringsize = popsize; % set offspring size
    offspring = zeros(popsize,DIM); %initialize auxilary data structure
    
    fm=min(10, size(sample_x,2));
    fm = 5;
%     t0 = 5 ; t1 = 50 ;
%     td = (maxfunevals-size(sample_x,1))/100;
%     fm = 2*(t1-t0)/(1+exp((td)))+t0;
     sample_y=feval(fun_name,sample_x); aa=sort(sample_y); bb=aa(floor(size(sample_x,1)/fm));
    
    %bound
    XVmin = repmat(XVmin, popsize,1);
    XVmax = repmat(XVmax, popsize,1);
    
    % Setup restarts
    stuckcount = 0;
    minvarcondition = setDE.minvarcondition;          % minimum variance which don't restart the algorithm
    stuckcond_noImp = setDE.stuckcond_noImp;          % maximum interations without improvement after which algorithm is restarted
    stuckcond_lowVar = setDE.stuckcond_lowVar;        % max. iter without improvement + low diversity condition
    
    %crossover_op = setDE.crossover_op;   % type of mutation operator: ['bin' or 'exp']
    %mutation_op = setDE.mutation_op;   % type of mutation operator: ['rand' or 'best' or 'average']
    
    F = 0.5;             % set parameter
    CR = 0.8; %setDE.CR;      % set cross-over probability
    
    statsDE = setDE.plotstatsDE;   % show variance in population, best solution and time per generations in a graph at the end of every trial
    statsJA = setJA.plotstatsJA;   % show mu_CR, mu_F, p_AE in a graph at the end of every trial
    
    JA_pArch = setJA.JA_pArch;          % how much best individuals to store (percentage, 0 = switch off)
    JA_pMut = setJA.JA_pMut;            % from 100*JA_pMut% of best ind. are randomly picked ind. in JA-mutation
    % (percentage, recommanded same as p_pArch but its min. value is fixed to 1 individual)
    JA_c_CR = setJA.JA_c_CR;    % mu_CR learning rate
    JA_c_F = setJA.JA_c_F;      % mu_F learning rate
    
    pop_de = initial_pop; %10 * rand(popsize, DIM) - 5;  % initialize population randomly within [-5, 5] in every dimension  
    popfit = feval(FUN, pop_de);
    [popfit, index] = sort(popfit);        % sort population
    pop_de = pop_de(index,:);              % sort population
    fbest = popfit(1);
    fbestold = fbest;
    
    maxiter = maxfunevals;% ceil( (maxfunevals-popsize)/popsize );% 
    
    % init archive - if nBestToArchive = 0 - then archive is switched off
    nBestToArchive = min(max(0, ceil(popsize * JA_pArch)), popsize);
    archive = Ar;
    auxArchive = [];
    
    % init JA-mutation
    adaptJA_muF = setJA.adaptJA_muF;    % use adaptation of mu_F
    adaptJA_muCR = setJA.adaptJA_muCR;  % use adaptation of mu_CR
    
    mu_F = setJA.JA_init_mu_F;    % init value for mu_F
    mu_CR = setJA.JA_init_mu_CR;	% init value for mu_CR
    
    nBestToMutate = min(max(1, ceil(popsize * JA_pMut)), popsize);
    randvectCR = ones(1,popsize)*mu_CR;
    randvectF = ones(1,popsize)*mu_F;
    
    if statsDE
        stat_var = zeros(1,maxiter);
        stat_best = zeros(1,maxiter);
        stat_time = zeros(1,maxiter);
        tic
    end
    
    if (statsJA == true)
        stat_muCR = zeros(1, maxiter+1);
        stat_muF = zeros(1, maxiter+1);
        stat_pAE = zeros(1, maxiter+1);
        
        stat_muCR(1) = mu_CR;
        stat_muF(1) = mu_F;
    end
    
    % 
    if rand(1)<0.5
        mutation_op='rand-A';
        action='case1';
        crossover_op='bin';
        F=0.5;
    else
        mutation_op='rand-1';
        action='case2';
        crossover_op='bin';
        CR = 1;
    end
        
    for iter = 1:maxiter
        
        isEncoded = zeros(popsize, 1);
        pop_en = pop_de;
        
        % MUTATION - rotationaly independent
        switch mutation_op
            case 'rand'
                randvectMU = ceil(rand(3,popsize)*popsize);   % may improve when applaying different numbers in every column
                for i = 1:popsize
                    %F = rand/2 + 0.5; % randomize F for every vector within range [0.5, 1]
                    offspring(i,:) = pop_en(randvectMU(1,i),:) + F *( pop_en(randvectMU(2,i),:) - pop_en(randvectMU(3,i),:) );
                end
            case 'rand-A'
                randvectMU = ceil(rand(2,popsize)*popsize);   % may improve when applaying different numbers in every column
                archiveCount = size(archive,1);
                sizePandA = popsize + archiveCount;
                randvectR1 = ceil(rand(1,popsize)*sizePandA); 
                aux = [pop_en; archive];
                randvectR2 = ceil(rand(1,popsize)*archiveCount); 
                for i = 1:popsize
                    %F = rand/2 + 0.5; % randomize F for every vector within range [0.5, 1]
                    offspring(i,:) = pop_en(randvectMU(1,i),:) + F *( pop_en(randvectMU(2,i),:) - aux(randvectR1(1,i),:) );
                end
            case 'rand-1'
                randvectMU = ceil(rand(3,popsize)*popsize);   % may improve when applaying different numbers in every column
                for i = 1:popsize
                    %F = rand/2 + 0.5; % randomize F for every vector within range [0.5, 1]
                    offspring(i,:) = pop_en(i,:) + F *( pop_en(randvectMU(1,i),:) - pop_en(i,:) )+ F *( pop_en(randvectMU(2,i),:) - pop_en(randvectMU(3,i),:) );
                end
            case 'best'
                randvectMU = ceil(rand(2,popsize)*popsize);
                randvectMU(2,randvectMU(1,:) ==randvectMU(2,:)) = mod(randvectMU(2,randvectMU(1,:) ==randvectMU(2,:))+1,popsize);
                randvectMU(2,randvectMU(2,:)==0) = popsize;
                for i = 1:popsize
                    %F = rand/2 + 0.5; % randomize F for every vector within range [0.5, 1]
                    offspring(i,:) = pop_en(1,:) + F *( pop_en(randvectMU(1,i),:) - pop_en(randvectMU(2,i),:) );
                end
            case 'average'
                avg_ind = pop_en*mut_weigh;
                
                randvectMU = ceil(rand(2,popsize)*popsize);
                randvectMU(2,randvectMU(1,:) ==randvectMU(2,:)) = mod(randvectMU(2,randvectMU(1,:) ==randvectMU(2,:))+1,popsize);
                randvectMU(2,randvectMU(2,:)==0) = popsize;
                for i = 1:popsize
                    %F = rand/2 + 0.5; % randomize F for every vector within range [0.5, 1]
                    offspring(:,i) = avg_ind + F *( pop_en(:,randvectMU(1,i)) - pop_en(:,randvectMU(2,i)) );
                end
            case 'JA'
                % random F
                generateSize = popsize*2;
                countInRange = 0;
                randvectF = zeros(1,popsize);
                for i = 1:50 % not probable to take so many loops
                    randc = 0.1 * tan(pi*(rand(1,generateSize)-0.5)) + mu_F;
                    randc = randc(randc > 0);
                    if (countInRange + length(randc) < popsize)
                        randvectF(countInRange + 1 : countInRange + length(randc)) = randc;
                        countInRange = countInRange + length(randc);
                    else
                        randvectF((countInRange+1) : popsize) = randc(1:popsize-countInRange );
                        break;
                    end
                end
                randvectF(randvectF > 1) = 1;
                randvectF=randvectF';
                F = repmat(randvectF, 1, DIM);
                
                % random R1 and R2
                sequence = 1:popsize;
                
                randvectR1 = ceil(rand(1,popsize) * popsize);
                randvectR1(randvectR1 == sequence) = mod(randvectR1(randvectR1 == sequence) + 1, popsize);
                randvectR1(randvectR1 == 0) = popsize;
                
                archiveCount = size(archive,1);
                sizePandA = popsize + archiveCount;
                randvectR2 = ceil(rand(1,popsize) * sizePandA);
                randvectR2(randvectR2 == sequence) = mod(randvectR2(randvectR2 == sequence) + 1, sizePandA);
                randvectR2(randvectR2 == 0) = sizePandA;
                randvectR2(randvectR2 == randvectR1) = mod(randvectR2(randvectR2 == randvectR1) + 1, sizePandA);
                randvectR2(randvectR2 == 0) = sizePandA;
                
                % random from nBestToMutate individuals from population
                randvectB = randi([1, nBestToMutate], 1, popsize);
                
                % append archive
                if(archiveCount > 0)
                    aux = [pop_en; archive(1:archiveCount,:)];
                else
                    aux = pop_en;
                end
                
                % create donors
                offspring = pop_en + F .* ( pop_en(randvectB,:) - pop_en ) + F .* (pop_en(randvectR1,:) - aux(randvectR2,:));
        end      
        
        % CROSS-OVER - rotationaly dependent
        switch crossover_op
            case 'bin'
                randvect = rand(popsize,DIM);
                offspring(randvect > CR) = pop_en(randvect > CR);    % new element from offspring is accepted when rand(0,1) < CR
            case 'exp'
                for i = 1:popsize
                    beginpos = randi([1,DIM], 1);
                    maxL = randi([1,DIM], 1);
                    pbbL = rand(maxL, 1);
                    L = find(pbbL>CR);
                    if isempty(L)
                        L = maxL;
                    else
                        L = L(1);
                    end
                    uvector = pop_en(:,i);
                    uvector(beginpos:min(beginpos+L-1,DIM)) = offspring(beginpos:min(beginpos+L-1,DIM),i);
                    uvector(1:mod(beginpos+L-1,DIM)) = offspring(1:mod(beginpos+L-1,DIM),i);
                    offspring(:,i) = uvector;
                end
            case 'JA'
                randvectCR = mu_CR + 0.1*randn(1,popsize);
                randvectCR(randvectCR < 0) = 0;
                randvectCR(randvectCR > 1) = 1;
                randmatCR2 = repmat(randvectCR, DIM, 1);
                randmatCR = rand(DIM, popsize);
                randmatCR = randmatCR < randmatCR2;
                
                randvectJ  = randi(DIM,1,popsize);
                randmatJ = zeros(DIM, popsize);
                randmatJ(sub2ind(size(randmatJ),randvectJ,1:popsize)) = 1;
                
                randmatCR=randmatCR'; randmatJ=randmatJ';
                
                offspring(~(randmatCR | randmatJ)) = pop_en(~(randmatCR | randmatJ));
            case 'non'
                %don't do crossover
                
        end
        
        % correcting violations on the lower bounds of the variables
        % these are good to go
        maskLB = offspring > XVmin;
        % these are good to go
        maskUB = offspring < XVmax;
        offspring     = offspring.*maskLB.*maskUB + XVmin.*(~maskLB) + XVmax.*(~maskUB);
        
        % REFINE
        switch action
            case 'case1'
            best_y = srgtsRBFEvaluate(offspring, srgtSRGT); 
            [~,b]=sort(best_y);
            for i=1:size(b,1)
                ds=offspring(b(i),:)-sample_x; ds=sum(ds.^2,2).^0.5;
                if min(ds)>1e-3, break; end 
            end
            if i~=size(b,1), sample_x=[sample_x;offspring(b(i),:)]; sample_y=feval(fun_name,sample_x); end
            srgtSRGT = Trainrbf(sample_x,sample_y);
            
            case 'case2'
            for i=1:size(offspring,1)
                ds=offspring(i,:)-sample_x; ds=sum(ds.^2,2).^0.5;
                check(i,:)=-min(ds);
            end
            [~,b]=sort(check);
            for i=1:size(b,1)
                ds=offspring(b(i),:)-sample_x; ds=sum(ds.^2,2).^0.5;
                if min(ds)>1e-3, break; end 
            end
            if i~=size(b,1), sample_x=[sample_x;offspring(b(i),:)]; sample_y=feval(fun_name,sample_x); end
            srgtSRGT = Trainrbf(sample_x,sample_y);
        end
        
        offspringfit = feval(FUN,offspring);
        % SELECTION
        % find better individuals
        ind = find(popfit>offspringfit);
        ind = find(offspringfit < bb | popfit>offspringfit); 
        
        % save into archive
        if (nBestToArchive > 0 && ~isempty(ind))
            archiveCount = size(archive,1);
            auxArchive(1:(archiveCount + length(ind)),:) = [archive; pop_de(ind,:)];
            
            p = randperm(archiveCount + length(ind));
            p = p(1:min(nBestToArchive, archiveCount + length(ind)));
            
            archive = auxArchive(p,:);
        end
        
        % overwrite by better individuals from offspring
        pop_de(ind,:) = offspring(ind,:);
        popfit(ind) = offspringfit(ind);
        
        % ADAPT ADAPTABLES
        if (adaptJA_muCR)
            meanA = mean(randvectCR(ind));
            if(~isnan(meanA))
                mu_CR = (1-JA_c_CR)*mu_CR + JA_c_CR*meanA;
            end
        end
        if (adaptJA_muF)
            meanL = sum(sum(randvectF(ind).^2))./sum(sum(randvectF(ind)));
            if(~isnan(meanL))
                mu_F = (1-JA_c_F)*mu_F + JA_c_F*meanL;
            end
        end
        
        % ORDER POPULATION BELONG FITNESS - because of adaptive encoding
        %                                   and average mutation
        [popfit, index] = sort(popfit);
        pop_de = pop_de(index,:);
        fbest = popfit(1);
        
        % STOPPING CRITERIA
        % interim stopping (algorithm is stucked in local optima)
        if fbest < fbestold
            % Improvement detected
            stuckcount = 0;
            fbestold = fbest;
        else
            stuckcount = stuckcount + 1;
        end
        %interim stopping (too low diversity)
        if (stuckcount > stuckcond_lowVar) && (sum(var(pop_de,1,2))/DIM < minvarcondition)
            %    if (max(std(pop_de,1,2)) < minvarcondition)
            exitcode = 'low variance';
            break;
        end
        % ftarget is reached
%         if feval(FUN, 'fbest') < ftarget  % task achieved
%             exitcode = 'solution found';
%             break;
%         end
        
        % DEBBUG AND STATS
        if statsDE
            stat_var(iter) = sum(var(pop_de,0,2));
            stat_best(iter) =fbest-ftarget; %feval(FUN, 'fbest') - ftarget;
            stat_time(iter) = toc;
        end
        
        if statsJA
            stat_muCR(iter+1) = mu_CR;
            stat_muF(iter+1)  = mu_F;
        end
        
    end
    
    %x = feval(FUN, 'fbest');
    
    if statsDE
        figure(1);
        plot(stat_var(1:iter-1));
        title('Variance in population');
        figure(2);
        semilogy(stat_best(1:iter-1));
        %plot(stat_best(1:iter-1));
        titstr = strcat ('Best found solution: ', num2str(fbest- ftarget));%num2str(feval(FUN, 'fbest') - ftarget));
        title(titstr);
        figure(3);
        %semilogy(stat_time(1:iter-1));
        plot(stat_time(1:iter-1));
        title('Computation time')
        
        %input('Press space to continue: ');
        %pause(1);
    end
    
    if statsJA
        figure(4);
        plot(stat_muCR(1:iter),'r');
        hold on
        title('Development of adaptation: mu_{CR} mu_{F}');
        plot(stat_muF(1:iter),'g');
        %plot(stat_pAE(1:iter),'b');
        legend('mu_{CR}', 'mu_{F}');
        hold off
        %input('Press space to continue: ');
        %pause(0);
    end
