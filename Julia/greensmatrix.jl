using LinearAlgebra
using Plots

# Implements eq 4 and prose around it in doi:10.1364/ol.41.001933 
function calculate_gij(ri, rj, k)
    rij = rj - ri
    nrij = norm(rij)
    rij_hat = normalize(rij)
    krij = k * nrij
    term1 = I - kron(rij_hat', rij_hat)
    term2 = 1 / (krij * 1im) + 1 / (krij)^2
    term3 = I - 3.0 * kron(rij_hat', rij_hat)
    @. -exp(1im * krij) / (1im * 4pi * nrij) * (term1 - term2 * term3)
end

function calculate_greens_matrix(p::AbstractArray, k)
    n = size(p, 1)
    G = zeros(ComplexF64, (3 * n, 3 * n))
    for i in range(2, n)
       for j in range(1, i-1)
           G[3 * i - 2:3 * i, 3 * j - 2:3 * j] = calculate_gij(p[i, 1:end], p[j, 1:end], k)
       end
    end
    G + transpose(G)
end

G = calculate_greens_matrix(a, sqrt(3));
vals, vecs = eigen(G);
scatter(real(vals), imag(vals))
