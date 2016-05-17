function [fobj]=linesearchRBF(eta)
% Fun��o objetivo para a otimiza��o em linha (LS)
% Retorna o erro de treinamento como fun��o da taxa de aprendizado
global nh nl np ent alvo weight bias ML MLant input target;
global gradp gradpant;
% C�pia tempor�ria das vari�veis do problema
weightcp=weight;
biascp=bias;
MLcp=ML;
MLantcp=MLant;
% Fun��o objetivo
eps=min(norm(gradp)^2/norm(gradpant)^2,epsm);
ML=gradp+eps*MLant;
aux=0;
fobj=0;
for j=1:nl
    for i=1:nh
        aux=aux+1;
        weight(i,j)=weight(i,j)-eta*ML(aux);
    end
    aux=aux+1;
    bias(j)=bias(j)-eta*ML(aux);
end
for k=1:np
    ent=input(k,:);
    alvo=target(k,:);
    [~,fob,~]=rbf();
    fobj=fobj+fob;
end

% Retorno das vari�veis aos valores originais
weight=weightcp;
bias=biascp;
ML=MLcp;
MLant=MLantcp;
end
