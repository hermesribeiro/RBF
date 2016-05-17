function [pattern]=geraLK(n)
% Gerador uniforme de pontos para o problema de Leonard-Kramer
% N: Número de pontos a serem gerados
% Gera n/3 pontos normais, n/3 falha 1 e n/3 falha 2
%
%  Fonte: Radial Basis Function Networks for Classifying Process Faults -
%  James A. Leonard, Mark A. Kramer - IEEE Control Systems - April 1991 -
%  Pgs. 31-38
pattern=zeros(n,5);
count=1;
% Pontos normais
while (count<=n/3)
    p1=normrnd(0,0.25);
    p2=normrnd(0,0.25);
    v1=normrnd(0,0.015);
    v2=normrnd(0,0.015);
    if (abs(p1)<0.05 && abs(p2)<0.05)
        pattern(count,1)=p1+p2+v1;
        pattern(count,2)=p1-p2+v2;
        pattern(count,3)=1;
        count=count+1;
    end
end
% Falha 1
while (count<=2*n/3)
    p1=normrnd(0,0.25);
    p2=normrnd(0,0.25);
    v1=normrnd(0,0.015);
    v2=normrnd(0,0.015);
    if (abs(p1)>0.05 && abs(p2)<0.05)
        pattern(count,1)=p1+p2+v1;
        pattern(count,2)=p1-p2+v2;
        pattern(count,4)=1;
        count=count+1;
    end
end
% Falha 2
while (count<=n)
    p1=normrnd(0,0.25);
    p2=normrnd(0,0.25);
    v1=normrnd(0,0.015);
    v2=normrnd(0,0.015);
    if (abs(p1)<0.05 && abs(p2)>0.05)
        pattern(count,1)=p1+p2+v1;
        pattern(count,2)=p1-p2+v2;
        pattern(count,5)=1;
        count=count+1;
    end
end
end