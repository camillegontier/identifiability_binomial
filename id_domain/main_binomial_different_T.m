% Solves equation 18 for sigma

%% Synaptic parameters

N = 4;
q = 1;
T_values = [50,100,200,300];
p_values = linspace(0,1,20);
k_gaussian = 2; %Number of parameters for M_0
k_binomial = 4; %Number of parameters for M_1

%% Loop over the values of p

identifiability_domain_1 = zeros(length(p_values),length(T_values));

iteration = 0;
for it_T = 1:length(T_values)
    T = T_values(it_T);
    for it_p = 1:length(p_values)
        iteration = iteration+1;
        disp('Iteration number' + string(iteration))

        p = p_values(it_p);

        BIC_difference_1 = @(sigma)abs(BIC_model_0(p,sigma,N,q,T,k_gaussian) - BIC_model_1(p,sigma,N,q,T,k_binomial));
        identifiability_domain_1(it_p,it_T) = fminbnd(BIC_difference_1,0,0.95);
    end
end

if p_values(1) == 0
    identifiability_domain_1(1,:) = 0;
end
if p_values(end) == 1
    identifiability_domain_1(end,:) = 0;
end

figure;
set(gca, 'ColorOrder', jet)
cc = winter(4);
hold on;
leg = '';

for it_T = 1:length(T_values)
    plot(identifiability_domain_1(:,it_T),p_values,'color',cc(it_T,:),'DisplayName','T = ' + string(T_values(it_T)),'LineWidth',3)
end

lgd = legend('Location','northwest');
lgd.NumColumns = 1;
grid on
xlabel('\sigma')
ylabel('p')

