#!/usr/bin/octave

format short 



%	Generating Data, 4 by 4 matrix with rank of 2
n = 100;
sampling_percentage = 0.5;
num_of_eig = 5;
seedM = floor(10*rand(n,n));
[Q,R] = qr(seedM);
V = Q(:,1:num_of_eig);
X = V*diag(1:num_of_eig)*V';
[U,S,V] = svd(X);
printf('Real eigen values %s\n' , mat2str(1:num_of_eig));
diag(S)




l = floor(n*sampling_percentage);
rp = randperm(n);
%rp = [1 3 2 4]

W = X(rp(1:l),rp(1:l));
G21 = X(rp(l+1:end), rp(1:l));
G22 = X(rp(l+1:end),rp(l+1:end));
C = [W;G21];


[V,D] = eig(X);
small_D = sort(diag(D),'descend');
small_D = small_D(1:num_of_eig+3)

[Vw,Dw] = eig(W);
small_Dw = (n/l)*sort(diag(Dw), 'descend');
small_Dw = small_Dw(1:num_of_eig)
