

%%Daqui para baixo e codigo antigo referente aos plots se der erro pode
%%cortar

for ii=1:6
    subplot(2,3,ii);
    [coef,melhor]=max(corrcoefs(:,ii));
    MM(ii,1:4)=[ii,seldegs(melhor,ii),coef,SamplesSizes(melhor)];
    MF(ii,:)=sblFos(melhor,ii,:);
    MT(ii,:)=sblTos(melhor,ii,:);
    T(1:3,ii)= [avgs(melhor,ii),stds(melhor,ii),covs(melhor,ii) ];
    fprintf("\n For QoI %d - Best emulator was n %d \n",ii,melhor);
    fprintf("Sobol")
    firstorder=sblFos(melhor,ii,:);
    Torder=sblTos(melhor,ii,:);
    bar(reshape(Torder,[1,8]),'b');
    hold on
    bar(reshape(firstorder,[1,8]),'g');
    hold off
    fprintf(" \n");
    legend( 'Total Order','First order');
end
saveas(gcf,folder+"sobol.png")


for ii = 1:6
    subplot(2,3,ii);
    [coef,melhor]=max(corrcoefs(:,ii));
    fprintf("\n For qoi %d, best emulator was n %d \n",ii,melhor);
    fprintf("Sobol")
    firstorder=sblFos(melhor,ii,:);
    Torder=sblTos(melhor,ii,:);
    hist(valR(melhor,:,ii));
    M(:,ii)=valR(melhor,:,ii);
    fprintf(" \n");
end

saveas(gcf,folder+"hist.png")




fprintf("Fitting and validation completed for all Sample Sizes! \n");
fprintf("Sample Sizes used \n");
disp(SamplesSizes)
fprintf(" \n");
fprintf("Selected degrees  \n");
disp(seldegs)
fprintf(" \n");
fprintf("Corr Coefs \n");
disp(corrcoefs)
fprintf("Avarage \n")
disp(avgcoefs);
fprintf(" \n");
fprintf("Min Errors \n");
disp(minEs)
fprintf(" \n");
fprintf("Max  Errors \n");
disp(maxEs)
fprintf(" \n");
fprintf("Mean Errors\n");
disp(meansE)
fprintf(" \n");

clf
figure('Position', [100, 100, 800, 600])

% Plot metrics
for ii = 1:6
   subplot(2,3,ii);
   yyaxis left

   MD(:,ii)=seldegs(:,ii);
   MC(:,ii)=corrcoefs(:,ii);

   plot(SamplesSizes,corrcoefs(:,ii))
   axis([ 0 NMAX+10 0.2  1.1]);
   ylabel("Avg Rsqr");
   hold on
   
   yyaxis right
   
   plot(SamplesSizes,seldegs(:,ii))
   axis([ 0 NMAX+10 0  7]);
   ylabel("Selected degree");
   xlabel("Training set size")
   for kk = 3:4
       for jj=1:4
        %xline(factorial(8+kk)/(factorial(8)*factorial(kk))*jj,'--b',label='D'+kk+" m"+jj,fontsize=6,LabelHorizontalAlignment='left')
        xline(factorial(8+kk)/(factorial(8)*factorial(kk))*jj,'--b'); %,fontsize=6);
       end
   end
   hold off
end 

SA=SamplesSizes;

saveas(gcf,folder+ "metrics.png")


clf
figure('Position', [100, 100, 800, 600])

% Plot metrics
for ii = 1:6
   subplot(2,3,ii);
   
   yyaxis left
   
   plot(SamplesSizes,meansE(:,ii))
   axis([ 0 NMAX -0.01  1.1*max(meansE(:,ii))]);
   ylabel("Mean Err");
   hold on
   
   yyaxis right

   plot(SamplesSizes,seldegs(:,ii))
   axis([ 0 NMAX 0  7]);
   ylabel("Selected degree");
   xlabel("Training set size")
   
   hold off
end 

saveas(gcf,folder+"meanerr.png")
clf

% Choose best emulator for each metric and perfom aditional validation 
for ii = 1:6
    subplot(2,3,ii);
    [coef,melhor]=max(corrcoefs(:,ii));
    fprintf("\n For qoi %d, best emulator was n %d \n",ii,melhor);
    fprintf("Generated using %d samples, and basis of %d degree \n",SamplesSizes(melhor),seldegs(melhor,ii));
    fprintf("Coef %f Mean Err %f \n",coef,meansE(melhor,ii));
    fprintf(" \n");
    val=ValSet(:,8+ii); %% dislocate 8 for inputs
    pred=valR(melhor,:,ii);
    
    scatter(pred,val,60,'filled');
    axis([ min(val) max(val) min(val)  max(val) ]);
    hold on
    plot(val,val,'black','LineWidth',2);
    hold off
    xlabel("Ytrue");
    ylabel("Ypred");
end 
saveas(gcf,folder+"scatter.png")
