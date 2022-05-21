using Random
using Statistics
using Plots

function V(x)
    x^2
end

function dmc()
    rng = Xoshiro()
    n_target = 800
    dt::Float32 = 10 / n_target
    t::Float32 = 0.0

    etrial::Float32 = 0.7
    e0::Float32 = 0.7
    energies = Float32[]
    nsweeps = 1_0_000
    sizehint!(energies, nsweeps)
    walker_positions = randn(rng, Float32, n_target)
    walker_positions_new = Float32[]
    q = similar(walker_positions)
    sizehint!(walker_positions, 2 * n_target)
    sizehint!(walker_positions_new, 2 * n_target)
    sizehint!(q, 2 * n_target)
    rand_cache = similar(walker_positions)
    for i = 1:nsweeps
        t += dt
        randn!(rng,rand_cache) #implace randn
        push!(energies, etrial)
        rand_cache .*= sqrt(dt)
        walker_positions .+= rand_cache # += randn(rng, Float32, length(walker_positions)) * sqrt(dt)
        @. q = exp(-dt * (V(walker_positions) - etrial))
        
        #rand_cache for s
        rand!(rng,rand_cache)
        rand_cache .+= q 
        s = rand_cache
        
        #note 2
        Δlength = 0
        @inbounds for si in eachindex(s)
            s_si = s[si]
            if s_si > 1.0
                iter = range(1+Δlength, Int(floor(s_si))+Δlength)
                walker_si = walker_positions[si]
                for _ in iter
                    Δlength += 1
                    walker_positions_new[Δlength] = walker_si
                end
            end
        end
        #resizing

        resize!(walker_positions,Δlength) 
        resize!(walker_positions_new,Δlength)
        resize!(q,Δlength)
        resize!(rand_cache,Δlength)

        i % 1000 == 0 && println("NW:", length(walker_positions_new), " ", etrial)
        pop_update_term = log(1f0 * n_target / length(walker_positions_new)) #note 3
        etrial = e0 + 0.1f0 * pop_update_term #note 4
        
        #copying
        walker_positions .= walker_positions_new      
    end
    energies
end

energies = dmc()
plot(energies)
histogram(energies)
