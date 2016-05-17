function [sai,erro,grad]=rbf();
global nh nl ent alvo c sigma weight bias;
% Sa�das, erro por padr�o e gradiente de uma rede RBF
s1=ent; % Sa�da da primeira camada
s2=zeros(1,nh);
% C�LCULO DA SA�DA DA CAMADA ESCONDIDA (RBF)
for k=1:nh
    dist=norm(s1-c(k,:));
    s2(k)=exp(-dist^2/sigma(k));    
end
% C�LCULO DA SA�DA DA CAMADA FINAL (SOFTMAX)
lambda=zeros(1,nl);
s3=zeros(1,nl);
for j=1:nl
   for k=1:nh
       lambda(j)=lambda(j)+weight(k,j)*s2(k);
   end
   lambda(j)=lambda(j)+bias(j);
   s3(j)=ativ(lambda(j));
end
s3=softmax(s3);
% C�LCULO DOS GRADIENTES DO ERRO EM RELA��O AS PESOS E VIESES
ndim=nh*nl+nl;
grad=zeros(1,ndim);
aux=0;
erro=0;
for j=1:nl
    t=alvo(j);
    s=s3(j);
    for i=1:nh
        aux=aux+1;
        grad(aux)=dpena(s,t)*dativ(s)*s2(i);
    end
    aux=aux+1;
    grad(aux)=dpena(s,t)*dativ(s);
    erro=erro+pena(s,t);
end
sai=s3;
sai=winner(s3); % P�s processamento: winner takes all
end

function [y]=pena(s,t)
% Fun��o erro (penalty function)
%y=(s-t)^2/2; %Soma dos quadrados
y=-((1-t)*log(1-s+1e-20)+t*log(s+1e-20));% Entropia cruzada
end
function [y]=dpena(s,t)
%C�lculo da derivada da fun��o erro
%y=s-t; %Ssoma dos quadrados
y=-(t-s); % Entropia cruzada
end
function [y]=ativ(lambda)
%C�lculo da fun��o de ativa��o
%y=1/(1+exp(-lambda)); % Log�stica
y=exp(lambda); %Calculo dos numeradores da softmax (chamar softmax(s3)!)
end
function [y]=dativ(s)
%Apenas com softmax
%C�lculo da derivada da fun��o de ativa��o
%y=s*(1-s); %Log�stica
y=1; %Softmax (Derivada j� inclu�da na derivada fun��o custo)
end
function [y]=softmax(x)
soma=sum(x);
y=x/soma;
end
function [y]=winner(x)
[~,pos]=max(x);
y=zeros(1,length(x));
y(pos)=1;
end