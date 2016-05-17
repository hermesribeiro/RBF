% Programa para cálculo de redes RBF (Ex. Leonard-Kramer problem)
% Autor: Hermes Ribeiro Sant' Anna
%
% Camada escondida: RBF
% Médoto de clusterização: K-means
% Cálculo das variâncias: P-nearest (P=2)
% Camada de saída: Softmax
% Função erro (ou penalidade): Entropia Cruzada
% Treinamento: Gradiente Conjugado com Busca em Linha (CG-LS)
%
% Referência principal: D.R. Baughman, Y.A. Liu, Neural Networks in 
% Bioprocessing and Chemical Engineering, 1995, Pgs 130-136.
% Referência secundária: 
% http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/
%
% Licenciado sob: FQQ-NR (Faça o Que Quiser - Não me Responsabilize)
clear
close
drawnow
global ni nh nl np ent alvo c sigma weight bias ML MLant input target;
global  gradp gradpant;
ni=2; % Número de variáveis de entrada
nh=24; % Número de neurônios escondidos
nl=3; % Número de variáveis de saída
nlvl=5; % Numero de níveis de aprendizado da camada RBF
nloops1=10; % Número de loops por nível
alpha=0.3; % Taxa de aprendizado inicial da RBF
nloops2=50; % Loops de aprendizado da camada final
etam=0.01; %Taxa mínima de aprendizado da camada final
epsM=1; %Momentum máximo
%PADRÕES PARA TREINO
np=150; % Número de padrões para treinamento
pattern=geraLK(np); % Gera pontos para o treinamento da rede
input=pattern(:,1:ni); % Carrega as variáveis de entrada na rede
target=pattern(:,ni+1:ni+nl); % Carrega as variáveis alvo
% PADRÕES PARA VALIDAÇÃO (TESTE, DEPENDENDO DA NOMENCLATURA)
npv=30; % Número de padrões para validação
patternv=geraLK(npv); % Gera pontos para validação da rede
inputv=patternv(:,1:ni); % Carrega os dados de entrada da validação
targetv=patternv(:,ni+1:ni+nl); % Carrega as variáveis alvo
% TREINAMENTO DA CAMADA RBF
c=sorteio(input,nh); % Esscolha de nh padrões como centro dos clusters
% Busca pelos centros dos clusters
tic
dist=zeros(1,nh); % Vetor auxiliar: Distância do centro os pontos
alpha0=alpha;
for lvl=1:nlvl; 
    alpha=alpha0*2^-(lvl-1); % Taxa de atualização dos centros
    for count=1:nloops1
        %Atualização dos centros
        for j=1:np
            for i=1:nh
                dist(i)=norm(input(j,:)-c(i,:)); % Distância euclideana
            end
            [~,perto]=min(dist); % Atualiza a posição do centro
            c(perto,:)=c(perto,:)+alpha*(input(j,:)-c(perto,:));
        end
    end
end
figure
plot(pattern(:,1),pattern(:,2),'o',c(:,1),c(:,2),'x');
drawnow
% Cálculo da variância (DP^2)
sigma=zeros(1,nh); % Desvio Padrão
for j=1:nh
    for i=1:nh
        dist(i)=norm(c(j)-c(i));
        if (i == j)
            dist(i)=1e9;
        end
    end
    [~,p1]=min(dist); % Centro mais próximo
    dist(p1)=1e10;
    [~,p2]=min(dist); % Segundo centro mais próximo
    sigma(j)=(0.5*(norm(c(i)-c(p1))^2+norm(c(i)-c(p2))^2))^0.5;
end
% APRENDIZADO POR GRADIENTE CONJUGADO
weight=-1+2*rand(nh,nl); % Inicialização dos pesos
bias=-1+2*rand(1,nl); % Inicialização dos viezes
sai=zeros(np,nl);
saiv=zeros(npv,nl);
ndim=nh*nl+nl;
gradpant=1e-5*ones(1,ndim); % Variável muda para entrada no loop
MLant=1e-5*ones(1,ndim); % Variável muda para entrada no loop
etaac=zeros(1,nloops2);
figure
for iter=1:nloops2
    gradp=zeros(1,ndim);
    fobjp(iter)=0; % Erro de cada época
    % Cálculo do erro e gradiente da época
    for k=1:np
        ent=input(k,:);
        alvo=target(k,:);
        [sai(k,:),fobj,grad]=rbf();
        fobjp(iter)=fobjp(iter)+fobj;
        gradp=gradp+grad;
    end
    % Atualização dos tensores de pesos e vieses
    eps=min(norm(gradp)^2/norm(gradpant)^2,epsM);
    ML=gradp+eps*MLant; % GC
    eta=fminbnd('linesearchRBF',etam,0.99); % LS
    %eta=etam; % Alternativa a LS (+velocidade -convergência)
    etaac(iter)=eta; % Rastro da taxa de apreodizado através das épocas
    aux=0;
    for j=1:nl
        for i=1:nh
            aux=aux+1;
            weight(i,j)=weight(i,j)-eta*ML(aux);
        end
        aux=aux+1;
        bias(j)=bias(j)-eta*ML(aux);
    end
    MLant=ML;
    gradpant=gradp;
    % Validação do treinamento(Early Stopping não Implementado)
    fobjv(iter)=0;
    for k=1:npv
        ent=inputv(k,:);
        alvo=targetv(k,:);
        [saiv(k,:),fobj,~]=rbf();
        fobjv(iter)=fobjv(iter)+fobj;
    end
    % Cálculo da taxa de acerto
    acerto=0;
    for i=1:np
        teste=mean(sai(i,:)==target(i,:));
        if (teste==1)
            acerto=acerto+1;
        end
        txacerto=acerto/np;
    end
    acertov=0;
    for i=1:npv
        teste=mean(saiv(i,:)==targetv(i,:));
        if (teste==1)
            acertov=acertov+1;
        end
        txacertov=acertov/npv;
    end
    ace(iter)=acerto;
    acev(iter)=acertov;
    txace(iter)=txacerto;
    txacev(iter)=txacertov;
    xx(iter)=iter;
    plot(xx,log10(fobjp),xx,log10(fobjv))
    %plot(xx,txace,xx,txacev)
    drawnow
end
acerto
txacerto
toc