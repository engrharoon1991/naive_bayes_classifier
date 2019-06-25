close all
clear all
clc
% Aprior Probability 
pwi=0.5;
pwj=0.5;
%Step No. 1
C=2; % No. of classes
d=2;    %No of features
s=200;  % No of Samples
mu=[5 8;-4 1];
mu1=mu(:,1); % mean of class 1 
mu2=mu(:,2);  %mean of class 2
%CASE 1 % Step No.2 & 3
sigma=[3 0;0 3]; % standerd Deviation
rng default  % For reproducibility 
class1 = mvnrnd(mu1,sigma,s)'; % producing multi varient random numbers
class2 = mvnrnd(mu2,sigma,s)';
% Step 4
% Plot Histogram of data
figure(1)
hist3(class1','Nbins',[15 15],'CDataMode','auto','FaceColor','interp')
hold on
hist3(class2','Nbins',[15 15])
xlim([-1 13]);
ylim([-10 5])
xlabel('Feature 1')
ylabel('Feature 2')
title('Histogram Plot of Orignal Data')
hold off
% Plot Gaussain with orignal mean and variance
% Density Function of Class 1
figure(2)
class1f1=class1(1,:)';
class1f2=class1(2,:)';
[X1,X2] = meshgrid(class1f1,class1f2); % creating grid
F = mvnpdf([X1(:) X2(:)],mu1',sigma);
F = reshape(F,length(class1f2),length(class1f1));
surf(class1f1,class1f2,F); % Three dimensional plot
caxis([min(F(:))-.5*range(F(:)),max(F(:))]);
%axis([-1 11 11 -11 0 1])
% Density Function of Class 2
hold on
class2f1=class2(1,:)';
class2f2=class2(2,:)';
[X3,X4] = meshgrid(class2f1,class2f2);
G = mvnpdf([X3(:) X4(:)],mu2',sigma);
G = reshape(G,length(class2f2),length(class2f1));
surf(class2f1,class2f2,G);
caxis([min(G(:))-.5*range(G(:)),max(G(:))]);
%axis([-1 11 11 -11 0 1])
title('Gussain Distribution Plot Sampled Data')
xlabel('Feature1'); ylabel('Feature 2'); zlabel('Probability Density');
hold off
%Step 5 and 6
class1Tr=class1(:,1:160);
class1Ts=class1(:,161:200);
class2Tr=class2(:,1:160);
class2Ts=class2(:,161:200);
%Step 7
class1Tr_noise = class1Tr + 0.5*randn(size(class1Tr)) + 0; % Mean =0 Std = 0.5
class2Tr_noise = class2Tr + 0.5*randn(size(class2Tr)) + 0; % Mean =0 Std = 0.5
% Step 8
% CASE NO. 1
%Calculating mean and varience of noisy data
mui=mean(class1Tr_noise,2);
muj=mean(class2Tr_noise,2);
%Calculating Covariance Matrix
sigma_n = cov(class1Tr_noise,class2Tr_noise);
%For case 1 classes are statistically independent no covariance exsist and variance of both matrix
%are same
sigma_case1 = mean(diag(sigma_n));
%Calculating W
W=(mui-muj);
W_t=W';
%Ploting all the data points and classifing it by line
figure(3)
%Step 9 Decision Boundry
scatter(class1Tr_noise(1,:),class1Tr_noise(2,:),'o', 'filled', 'b')
hold on
scatter(class2Tr_noise(1,:),class2Tr_noise(2,:),'o', 'filled', 'g')
X_0 = .5*(mui+muj)-(sigma_case1^2/norm(W))*log(pwi/pwj)*W;
X=pinv(W_t).*W_t*X_0
plot(X_0(1,1),X_0(2,1),'*')
plot(X(1,1),X(2,1),'+')
line([X(1,1) X_0(1,1)],[X(2,1) X_0(2,1)])
xlabel('Feature1'); ylabel('Feature 2');
title('Case I - Decision Boundry(STD=0.5)')
x = linspace(-1,15,100);
y =((X_0(2,1)-X(2,1))/(X_0(1,1)-X(1,1)))*(x-X(1,1))+X(2,1);
plot(x,y)
hold off
%Step 10
%Accuracy of Classifier 
% counti=0;
% countj=0;
% for n=1:40
% gpix=((-norm(class1Ts(:,n)-mui))^2)/(2*sigma_case1^2);
% gpjx=((-norm(class1Ts(:,n)-muj))^2)/(2*sigma_case1^2);
%  if (gpix>gpjx)
%      counti=counti+1;
%  else
%      countj=countj+1;
%  end
% end
% Accuracy_Case11=counti/40;
% 
% counti=0;
% countj=0;
% for n=1:40
% gpix=((-norm(class2Ts(:,n)-mui))^2)/(2*sigma_case1^2);
% gpjx=((-norm(class2Ts(:,n)-muj))^2)/(2*sigma_case1^2);
%  if (gpjx>gpix)
%      counti=counti+1;
%  else
%      countj=countj+1;
%  end
% end
% Accuracy_Case12=countj/40;

%Step 11 
% For STD = 1
class1Tr_noise = class1Tr + 1*randn(size(class1Tr)) + 0; % Mean =0 Std = 1
class2Tr_noise = class2Tr + 1*randn(size(class2Tr)) + 0; % Mean =0 Std = 1
mui=mean(class1Tr_noise,2)
muj=mean(class2Tr_noise,2)
sigma_n = cov(class1Tr_noise,class2Tr_noise)
sigma_case1 = mean(diag(sigma_n));
W=(mui-muj);
W_t=W';
figure(4)
scatter(class1Tr_noise(1,:),class1Tr_noise(2,:),'o', 'filled', 'b')
hold on
scatter(class2Tr_noise(1,:),class2Tr_noise(2,:),'o', 'filled', 'g')
X_0 = .5*(mui+muj)-(sigma_case1^2/norm(W))*log(pwi/pwj)*W;
X=pinv(W_t).*W_t*X_0;
plot(X_0(1,1),X_0(2,1),'*')
plot(X(1,1),X(2,1),'+')
line([X(1,1) X_0(1,1)],[X(2,1) X_0(2,1)])
xlabel('Feature1'); ylabel('Feature 2');
title('Case I - Decision Boundry(STD=1)')
x = linspace(-1,15,100);
y =((X_0(2,1)-X(2,1))/(X_0(1,1)-X(1,1)))*(x-X(1,1))+X(2,1);
plot(x,y)
hold off
% Accuracy of Classifier

% For STD = 1.5
class1Tr_noise = class1Tr + 1.5*randn(size(class1Tr)) + 0; % Mean =0 Std = 1.5
class2Tr_noise = class2Tr + 1.5*randn(size(class2Tr)) + 0; % Mean =0 Std = 1.5
mui=mean(class1Tr_noise,2)
muj=mean(class2Tr_noise,2)
sigma_n = cov(class1Tr_noise,class2Tr_noise)
sigma_case1 = mean(diag(sigma_n));
W=(mui-muj);
W_t=W';
figure(5)
scatter(class1Tr_noise(1,:),class1Tr_noise(2,:),'o', 'filled', 'b')
hold on
scatter(class2Tr_noise(1,:),class2Tr_noise(2,:),'o', 'filled', 'g')
X_0 = .5*(mui+muj)-(sigma_case1^2/(norm(W))^2)*log(pwi/pwj)*W;
X=pinv(W_t).*W_t*X_0
plot(X_0(1,1),X_0(2,1),'*')
plot(X(1,1),X(2,1),'+')
line([X(1,1) X_0(1,1)],[X(2,1) X_0(2,1)])
xlabel('Feature1'); ylabel('Feature 2');
title('Case I - Decision Boundry(STD=1.5)')
x = linspace(-1,15,100);
y =((X_0(2,1)-X(2,1))/(X_0(1,1)-X(1,1)))*(x-X(1,1))+X(2,1);
plot(x,y)
hold off
% CASE NO. 2 STD=0.5
%Step 12 Changing Sigma
sigma = [4 3;1 9];
%Step 13
class1Tr_noise = class1Tr + 0.5*randn(size(class1Tr)) + 0; % Mean =0 Std = 0.5
class2Tr_noise = class2Tr + 0.5*randn(size(class2Tr)) + 0; % Mean =0 Std = 0.5
%Step 14
%Calculating mean and varience of noisy data
mui=mean(class1Tr_noise,2)
muj=mean(class2Tr_noise,2)
%Calculating Covariance Matrix
sigma_n = cov(class1Tr_noise,class2Tr_noise);
W=pinv(sigma_n).*(mui-muj);
W_t=W';
figure(6)
scatter(class1Tr_noise(1,:),class1Tr_noise(2,:),'o', 'filled', 'b')
hold on
scatter(class2Tr_noise(1,:),class2Tr_noise(2,:),'o', 'filled', 'g')
X_0 = .5*(mui+muj)- (log(pwi/pwj).*(mui-muj).*pinv((mui-muj)'.*W.*(mui-muj)));
X=pinv(W_t).*W_t*X_0
plot(X_0(1,1),X_0(2,1),'*')
plot(X(1,1),X(2,1),'+')
xlabel('Feature1'); ylabel('Feature 2');
title('Case II - Decision Boundry where covariance is same')
%Step 15
x = linspace(-10,35,100);
y =((X_0(2,1)-X(2,1))/(X_0(1,1)-X(1,1)))*(x-X(1,1))+X(2,1);
plot(x,y)
hold off
% Step 16 
% Accuracy of Classifier
% Step 17
%STD=1
class1Tr_noise = class1Tr + 1*randn(size(class1Tr)) + 0; % Mean =0 Std = 1
class2Tr_noise = class2Tr + 1*randn(size(class2Tr)) + 0; % Mean =0 Std = 1
%Calculating mean and varience of noisy data
mui=mean(class1Tr_noise,2)
muj=mean(class2Tr_noise,2)
%Calculating Covariance Matrix
sigma_n = cov(class1Tr_noise,class2Tr_noise);
W=pinv(sigma_n).*(mui-muj);
W_t=W';
figure(7)
scatter(class1Tr_noise(1,:),class1Tr_noise(2,:),'o', 'filled', 'b')
hold on
scatter(class2Tr_noise(1,:),class2Tr_noise(2,:),'o', 'filled', 'g')
X_0 = .5*(mui+muj)- (log(pwi/pwj).*(mui-muj).*pinv((mui-muj)'.*W.*(mui-muj)));
X=pinv(W_t).*W_t*X_0
plot(X_0(1,1),X_0(2,1),'*')
plot(X(1,1),X(2,1),'+')
line([X(1,1) X_0(1,1)],[X(2,1) X_0(2,1)])
xlabel('Feature1'); ylabel('Feature 2');
title('Case II - Decision Boundry (STD=1)')
x = linspace(-10,25,100);
y =((X_0(2,1)-X(2,1))/(X_0(1,1)-X(1,1)))*(x-X(1,1))+X(2,1);
plot(x,y)
hold off
%STD=1.5
class1Tr_noise = class1Tr + 1.5*randn(size(class1Tr)) + 0; % Mean =0 Std = 1.5
class2Tr_noise = class2Tr + 1.5*randn(size(class2Tr)) + 0; % Mean =0 Std = 1.5
%Calculating mean and varience of noisy data
mui=mean(class1Tr_noise,2)
muj=mean(class2Tr_noise,2)
%Calculating Covariance Matrix
sigma_n = cov(class1Tr_noise,class2Tr_noise);
W=pinv(sigma_n).*(mui-muj);
W_t=W';
figure(8)
scatter(class1Tr_noise(1,:),class1Tr_noise(2,:),'o', 'filled', 'b')
hold on
scatter(class2Tr_noise(1,:),class2Tr_noise(2,:),'o', 'filled', 'g')
plot(X_0(1,1),X_0(2,1),'*')
plot(X(1,1),X(2,1),'+')
line([X(1,1) X_0(1,1)],[X(2,1) X_0(2,1)])
xlabel('Feature1'); ylabel('Feature 2');
title('Case II - Decision Boundry (STD=1.5)')
x = linspace(-10,20,100);
y =((X_0(2,1)-X(2,1))/(X_0(1,1)-X(1,1)))*(x-X(1,1))+X(2,1);
plot(x,y)
hold off
%Step 18
%CASE III Std =0.5
sigma_i = [4 3;1 9];
sigma_j = [5 4;2 8];
%Step 19
class1Tr_noise = class1Tr + 0.5*randn(size(class1Tr)) + 0; % Mean =0 Std = 0.5
class2Tr_noise = class2Tr + 0.5*randn(size(class2Tr)) + 0; % Mean =0 Std = 0.5
%Step 20
mui=mean(class1Tr_noise,2);
muj=mean(class2Tr_noise,2);
sigma_i_inv=inv(sigma_i);
sigma_j_inv=inv(sigma_j);
sigma_i_d=det(sigma_i);
sigma_j_d=det(sigma_j);
Ai=-.5*(sigma_i_inv);
Aj=-.5*(sigma_j_inv)
Bi=(sigma_i_inv)*mui
Bj=(sigma_j_inv)*muj
Ci=(-0.5*(mui)'*(sigma_i_inv)*(mui))-(.5*log(sigma_i_d))% lnP(wi) and lnP(wi) are eqaul 
Cj=(-0.5*(muj)'*(sigma_j_inv)*(muj))-(.5*log(sigma_j_d))% lnP(wi) and lnP(wi) are eqaul 
An=Ai-Aj;
Bn=Bi-Bj;
Cn=Ci-Cj;
%Step 21
figure(9)
scatter(class1Tr_noise(1,:),class1Tr_noise(2,:),'o', 'filled', 'b')
hold on
scatter(class2Tr_noise(1,:),class2Tr_noise(2,:),'o', 'filled', 'g')
syms x1 x2
xn=[x1;x2];
xnt=xn';
eq=An(1,1)*x1^2+An(1,2)*x1*x2+An(2,1)*x1*x2+An(2,2)*x2^2+Bn(1,1)*x1+Bn(2,1)*x2+Cn
x1=solve(eq==0,x1);
Fx11=matlabFunction(x1(1));
Fx12=matlabFunction(x1(2));
fplot(Fx11)
title('Case III - Decision Boundry (STD=0.5)')
xlabel('Feature1'); ylabel('Feature 2');
hold on
fplot(Fx12)
hold off
%Step 23
% For Standerd Deviation of 1
class1Tr_noise = class1Tr + 1*randn(size(class1Tr)) + 0; % Mean =0 Std = 1
class2Tr_noise = class2Tr + 1*randn(size(class2Tr)) + 0; % Mean =0 Std = 1
mui=mean(class1Tr_noise,2);
muj=mean(class2Tr_noise,2);
sigma_i_inv=inv(sigma_i);
sigma_j_inv=inv(sigma_j);
sigma_i_d=det(sigma_i);
sigma_j_d=det(sigma_j);
Ai=-.5*(sigma_i_inv);
Aj=-.5*(sigma_j_inv)
Bi=(sigma_i_inv)*mui
Bj=(sigma_j_inv)*muj
Ci=(-0.5*(mui)'*(sigma_i_inv)*(mui))-(.5*log(sigma_i_d))% lnP(wi) and lnP(wi) are eqaul 
Cj=(-0.5*(muj)'*(sigma_j_inv)*(muj))-(.5*log(sigma_j_d))% lnP(wi) and lnP(wi) are eqaul 
An=Ai-Aj;
Bn=Bi-Bj;
Cn=Ci-Cj;
figure(10)
scatter(class1Tr_noise(1,:),class1Tr_noise(2,:),'o', 'filled', 'b')
hold on
scatter(class2Tr_noise(1,:),class2Tr_noise(2,:),'o', 'filled', 'g')
syms x1 x2
xn=[x1;x2];
xnt=xn';
eq=An(1,1)*x1^2+An(1,2)*x1*x2+An(2,1)*x1*x2+An(2,2)*x2^2+Bn(1,1)*x1+Bn(2,1)*x2+Cn
x1=solve(eq==0,x1);
Fx11=matlabFunction(x1(1));
Fx12=matlabFunction(x1(2));
fplot(Fx11)
title('Case III - Decision Boundry (STD=1)')
xlabel('Feature1'); ylabel('Feature 2');
hold on
fplot(Fx12)
hold off
% CASEIII: For Standerd Deviation of 1.5
class1Tr_noise = class1Tr + 1.5*randn(size(class1Tr)) + 0; % Mean =0 Std = 1
class2Tr_noise = class2Tr + 1.5*randn(size(class2Tr)) + 0; % Mean =0 Std = 1
mui=mean(class1Tr_noise,2);
muj=mean(class2Tr_noise,2);
sigma_i_inv=inv(sigma_i);
sigma_j_inv=inv(sigma_j);
sigma_i_d=det(sigma_i);
sigma_j_d=det(sigma_j);
Ai=-.5*(sigma_i_inv);
Aj=-.5*(sigma_j_inv)
Bi=(sigma_i_inv)*mui
Bj=(sigma_j_inv)*muj
Ci=(-0.5*(mui)'*(sigma_i_inv)*(mui))-(.5*log(sigma_i_d))% lnP(wi) and lnP(wi) are eqaul 
Cj=(-0.5*(muj)'*(sigma_j_inv)*(muj))-(.5*log(sigma_j_d))% lnP(wi) and lnP(wi) are eqaul 
An=Ai-Aj;
Bn=Bi-Bj;
Cn=Ci-Cj;
figure(11)
scatter(class1Tr_noise(1,:),class1Tr_noise(2,:),'o', 'filled', 'b')
hold on
scatter(class2Tr_noise(1,:),class2Tr_noise(2,:),'o', 'filled', 'g')
syms x1 x2
xn=[x1;x2];
xnt=xn';
eq=An(1,1)*x1^2+An(1,2)*x1*x2+An(2,1)*x1*x2+An(2,2)*x2^2+Bn(1,1)*x1+Bn(2,1)*x2+Cn
x1=solve(eq==0,x1);
Fx11=matlabFunction(x1(1));
Fx12=matlabFunction(x1(2));
fplot(Fx11)
title('Case III - Decision Boundry (STD=1.5)')
xlabel('Feature1'); ylabel('Feature 2');
hold on
fplot(Fx12)
hold off