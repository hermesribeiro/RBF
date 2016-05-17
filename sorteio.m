function [y]=sorteio(x,n)
% Dada uma matriz x e um número de sorteios n a função retorna uma matriz y
% com n linhas escolhidas aleatóriamente
%
[m,o]=size(x);
if (n>m)
    return;
end
y=zeros(n,o);
for i=1:n
    pick=floor(m*rand+1);
    temp=x(pick,:);
    x(pick,:)=x(m-i+1,:);
    x(m-i+1,:)=temp;
    y(i,:)=x(m-i+1,:);
end

end