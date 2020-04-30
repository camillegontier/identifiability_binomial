function BIC = BIC_model_1(p,sigma,N,q,T,number_parameters)

% Computes the average BIC for model M_1 (left-hand part of equation 18)

    function p_ln_p = integrant(x,sigma,p,N,q)
        p_e = 0;
        for k = 0:N
            p_e = p_e + normpdf(x,k*q,sigma)*binopdf(k,N,p);
        end
        p_ln_p = p_e*log(p_e);        
    end

% Born for numerical integration
born_sup = N*p*q + 4*sqrt(sigma^2 + q^2*N*p*(1-p));
born_inf = N*p*q - 4*sqrt(sigma^2 + q^2*N*p*(1-p));
BIC = -2*T*integral(@(x)integrant(x,sigma,p,N,q),born_inf,born_sup,'ArrayValued',1)+number_parameters*log(T);

end

