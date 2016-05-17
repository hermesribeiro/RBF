% Programa para c�lculo de redes RBF (Ex. Leonard-Kramer problem)
% Autor: Hermes Ribeiro Sant' Anna
%
% Camada escondida: RBF
% M�doto de clusteriza��o: K-means
% C�lculo das vari�ncias: P-nearest (P=2)
% Camada de sa�da: Softmax
% Fun��o erro (ou penalidade): Entropia Cruzada
% Treinamento: Gradiente Conjugado com Busca em Linha (CG-LS)
%
% Refer�ncia principal: D.R. Baughman, Y.A. Liu, Neural Networks in 
% Bioprocessing and Chemical Engineering, 1995, Pgs 130-136.
% Refer�ncia secund�ria: 
% http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/
%
% Licenciado sob: FQQ-NR (Fa�a o Que Quiser - N�o me Responsabilize)
clear
close
drawnow
global ni nh nl np ent alvo c sigma weight bias ML MLant input target;
global  gradp gradpant;
ni=2; % N�mero de vari�veis de entrada
nh=24; % N�mero de neur�nios escondidos
nl=3; % N�mero de vari�veis de sa�da
nlvl=5; % Numero de n�veis de aprendizado da camada RBF
nloops1=10; % N�mero de loops por n�vel
alpha=0.3; % Taxa de aprendizado inicial da RBF
nloops2=50; % Loops de aprendizado da camada final
etam=0.01; %Taxa m�nima de aprendizado da camada final
epsM=1; %Momentum m�ximo
%PADR�ES PARA TREINO
np=150; % N�mero de padr�es para treinamento
pattern=geraLK(np); % Gera pontos para o treinamento da rede
input=pattern(:,1:ni); % Carrega as vari�veis de entrada na rede
target=pattern(:,ni+1:ni+nl); % Carrega as vari�veis alvo
% PADR�ES PARA VALIDA��O (TESTE, DEPENDENDO DA NOMENCLATURA)
npv=30; % N�mero de padr�es para valida��o
patternv=geraLK(npv); % Gera pontos para valida��o da rede
inputv=patternv(:,1:ni); % Carrega os dados de entrada da valida��o
targetv=patternv(:,ni+1:ni+nl); % Carrega as vari�veis alvo
% TREINAMENTO DA CAMADA RBF
c=sorteio(input,nh); % Esscolha de nh padr�es como centro dos clusters
% Busca pelos centros dos clusters
tic
dist=zeros(1,nh); % Vetor auxiliar: Dist�ncia do centro os pontos
alpha0=alpha;
for lvl=1:nlvl; 
    alpha=alpha0*2^-(lvl-1); % Taxa de atualiza��o dos centros
    for count=1:nloops1
        %Atualiza��o dos centros
        for j=1:np
            for i=1:nh
                dist(i)=norm(input(j,:)-c(i,:)); % Dist�ncia euclideana
            end
            [~,perto]=min(dist); % Atualiza a posi��o do centro
            c(perto,:)=c(perto,:)+alpha*(input(j,:)-c(perto,:));
        end
    end
end
figure
plot(pattern(:,1),pattern(:,2),'o',c(:,1),c(:,2),'x');
drawnow
% C�lculo da vari�ncia (DP^2)
sigma=zeros(1,nh); % Desvio Padr�o
for j=1:nh
    for i=1:nh
        dist(i)=norm(c(j)-c(i));
        if (i == j)
            dist(i)=1e9;
        end
    end
    [~,p1]=min(dist); % Centro mais pr�ximo
    dist(p1)=1e10;
    [~,p2]=min(dist); % Segundo centro mais pr�ximo
    sigma(j)=(0.5*(norm(c(i)-c(p1))^2+norm(c(i)-c(p2))^2))^0.5;
end
% APRENDIZADO POR GRADIENTE CONJUGADO
weight=-1+2*rand(nh,nl); % Inicializa��o dos pesos
bias=-1+2*rand(1,nl); % Inicializa��o dos viezes
sai=zeros(np,nl);
saiv=zeros(npv,nl);
ndim=nh*nl+nl;
gradpant=1e-5*ones(1,ndim); % Vari�vel muda para entrada no loop
MLant=1e-5*ones(1,ndim); % Vari�vel muda para entrada no loop
etaac=zeros(1,nloops2);
figure
for iter=1:nloops2
    gradp=zeros(1,ndim);
    fobjp(iter)=0; % Erro de cada �poca
    % C�lculo do erro e gradiente da �poca
    for k=1:np
        ent=input(k,:);
        alvo=target(k,:);
        [sai(k,:),fobj,grad]=rbf();
        fobjp(iter)=fobjp(iter)+fobj;
        gradp=gradp+grad;
    end
    % Atualiza��o dos tensores de pesos e vieses
    eps=min(norm(gradp)^2/norm(gradpant)^2,epsM);
    ML=gradp+eps*MLant; % GC
    eta=fminbnd('linesearchRBF',etam,0.99); % LS
    %eta=etam; % Alternativa a LS (+velocidade -converg�ncia)
    etaac(iter)=eta; % Rastro da taxa de apreodizado atrav�s das �pocas
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
    % Valida��o do treinamento(Early Stopping n�o Implementado)
    fobjv(iter)=0;
    for k=1:npv
        ent=inputv(k,:);
        alvo=targetv(k,:);
        [saiv(k,:),fobj,~]=rbf();
        fobjv(iter)=fobjv(iter)+fobj;
    end
    % C�lculo da taxa de acerto
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