% Dynamic Time Warping
function [Dist,D,k,w,rw,tw]=dtw(r,t,pflag)

% [Dist,D,k,w,rw,tw]=dtw(r,t,pflag)
% Dynamic Time Warping Algorithm
% Dist is unnormalized distance between t and r
% D is the accumulated distance matrix
% k is the normalizing factor
% w is the optimal path
% t is the vector you are testing against
% r is the vector you are testing
% rw is the warped r vector
% tw is the warped t vector
% pflag  plot flag: 1 (yes), 0(no)
% Version comments:

[row,M]=size(r); 
if (row > M) 
    M=row; r=r';  
end

[row,N]=size(t); 
if (row > N) 
    N=row; t=t'; 
end

d=sqrt((repmat(r',1,N)-repmat(t,M,1)).^2); %得到两序列对比后的欧式距离矩阵

%sqrt(X)求X所有元素的平方根
%repmat()让r平铺1行N列的矩阵
%所以把r和t的值都撒入坐标矩阵中，相减后分别计算平方后的开方值，也就是欧式距离。

D=zeros(size(d));%按d的大小初始化一个零矩阵给累加矩阵D
D(1,1)=d(1,1);%令第一个累加值等于其欧式距离值，由于D（1，1）只有D（0，0）可以依靠，但D（0，0）=0，所以D(1,1)=0+d(1,1)

for m=2:M

    D(m,1)=d(m,1)+D(m-1,1);%在欧式距离矩阵中先计算第一列的每个累加距离
end

for n=2:N

    D(1,n)=d(1,n)+D(1,n-1);%在欧式距离矩阵中先计算第一行的每个累加距离
end

for m=2:M

    for n=2:N
    
        D(m,n)=d(m,n)+min(D(m-1,n),min(D(m-1,n-1),D(m,n-1))); % this double MIn construction improves in 10-fold the Speed-up.
    end
end

Dist=D(M,N);%把最终累加距离给Dist

%实际使用时到这里就可以结束了，只需要得到距离，而后面是可视化的内容

n=N;
m=M;
k=1;
w=[M N];%给出路径终点,开始反推路径

while ((n+m)~=2)
    if (n-1)==0
        m=m-1;
    elseif (m-1)==0
        n=n-1;
    else 
      [~,number]=min([D(m-1,n),D(m,n-1),D(m-1,n-1)]);
      switch number
      case 1
        m=m-1;
      case 2
        n=n-1;
      case 3
        m=m-1;
        n=n-1;
      end
    end
    
k=k+1;
    w=[m n; w]; %累加最优路径
end

% warped waves
rw=r(w(:,1));
tw=t(w(:,2));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if pflag

 % --- Accumulated distance matrix and optimal path

 figure('Name','DTW - Accumulated distance matrix and optimal path', 'NumberTitle','off');

main1=subplot('position',[0.19 0.19 0.67 0.79]);

 image(D);
    cmap = contrast(D);
    colormap(cmap); % 'copper' 'bone', 'gray' imagesc(D);
    hold on;
    x=w(:,1); y=w(:,2);
    ind= x==1; x(ind)=1+0.2;
    ind= x==M; x(ind)=M-0.2;
    ind= y==1; y(ind)=1+0.2;
    ind= y==N; y(ind)=N-0.2;
    plot(y,x,'-w', 'LineWidth',1);
    hold off;
    axis([1 N 1 M]);
    set(main1, 'FontSize',7, 'XTickLabel','', 'YTickLabel','');

colorb1=subplot('position',[0.88 0.19 0.05 0.79]);
    nticks=8;
    ticks=floor(1:(size(cmap,1)-1)/(nticks-1):size(cmap,1));
    mx=max(max(D));
    mn=min(min(D));
    ticklabels=floor(mn:(mx-mn)/(nticks-1):mx);
    colorbar(colorb1);
    set(colorb1, 'FontSize',7, 'YTick',ticks, 'YTickLabel',ticklabels);
    set(get(colorb1,'YLabel'), 'String','Distance', 'Rotation',-90, 'FontSize',7, 'VerticalAlignment','bottom');

left1=subplot('position',[0.07 0.19 0.10 0.79]);
    plot(r,M:-1:1,'-b');
    set(left1, 'YTick',mod(M,10):10:M, 'YTickLabel',10*rem(M,10):-10:0)
    axis([min(r) 1.1*max(r) 1 M]);
    set(left1, 'FontSize',7);
    set(get(left1,'YLabel'), 'String','Samples', 'FontSize',7, 'Rotation',-90, 'VerticalAlignment','cap');
    set(get(left1,'XLabel'), 'String','Amp', 'FontSize',6, 'VerticalAlignment','cap');

bottom1=subplot('position',[0.19 0.07 0.67 0.10]);
    plot(t,'-r');
    axis([1 N min(t) 1.1*max(t)]);
    set(bottom1, 'FontSize',7, 'YAxisLocation','right');
    set(get(bottom1,'XLabel'), 'String','Samples', 'FontSize',7, 'VerticalAlignment','middle');
    set(get(bottom1,'YLabel'), 'String','Amp', 'Rotation',-90, 'FontSize',6, 'VerticalAlignment','bottom');

 % --- Warped signals

 figure('Name','DTW - warped signals', 'NumberTitle','off');

 subplot(1,2,1);
    set(gca, 'FontSize',7);
    hold on;
    plot(r,'-bx');
    plot(t,':r.');
    hold off;
    axis([1 max(M,N) min(min(r),min(t)) 1.1*max(max(r),max(t))]);
    grid;
    legend('signal 1','signal 2');
    title('Original signals');
    xlabel('Samples');
    ylabel('Amplitude');

subplot(1,2,2);
    set(gca, 'FontSize',7);
    hold on;
    plot(rw,'-bx');
    plot(tw,':r.');
    hold off;
    axis([1 k min(min([rw; tw])) 1.1*max(max([rw; tw]))]);
    grid;
    legend('signal 1','signal 2');
    title('Warped signals');
    xlabel('Samples');
    ylabel('Amplitude');
end