using GenericLinearAlgebra
using SymPy
using PyCall
np = pyimport("numpy")

function create_rho_list(levels = 3)
    rho_list = []
    for i in 1:levels
        for j in 1:levels
            push!(rho_list,"rho_$(i)$(j)")
        end
    end
    return rho_list #!Revisar lo de la asignación con las variables globales
end
        
function def_rho_matrix(levels)
    rho_matrix = Matrix{Sym}(undef,levels,levels)
    for i in 1:levels
        for j in 1:levels      
            rho_matrix[i, j] = symbols("rho_$(i)$(j)")
        end
    end #!Revisar lo de la asignación con las variables globales
    #println(rho_matrix)
    #println(rho_matrix[1,1])  #Julia comienza a contar en 1, no en 0 como python.
return np.matrix(rho_matrix)
end

function Hamiltonian(Omegas, Deltas)
    #Given lists of Rabi frequencies and detunings, construct interaction 
    """
    Hamiltonian (assuming RWA and dipole approximation) as per equation 14. 
    h_bar = 1 for simplicity.
    Both lists should be in ascending order (Omega_12, Omega_23 etc) """
    levels = length(Omegas)+1
    H = np.zeros((levels,levels))
    for i in 1:levels
        for j in 1:levels
            if i==j && i!=1
                H[i,j] = -2*(np.sum(Deltas[1:i-1]))
        
            elseif abs(i-j) == 1
                H[i,j] = Omegas[np.min([i,j])]
            end
        end
    end
    return np.matrix(H/2)
end

function L_decay(Gammas)
    """
    Given a list of linewidths for each atomic level, 
    construct atomic decay operator in matrix form for a ladder system.
    Assumes that there is no decay from the lowest level (Gamma_1 = 0).
    """
   
    levels = length(Gammas)+1
    rhos = def_rho_matrix(levels)
    Gammas_all = [0; Gammas] #Adds a zero because there is no decay from the lowest level
    decay_matrix = np.zeros((levels,levels), dtype = "object")
    
    for i in 1:levels
        for j in 1:levels
           
            if i != j
                decay_matrix[i,j] = -0.5*(
                    Gammas_all[i]+Gammas_all[j])*rhos[i,j]
                    
            elseif i != levels
                println(i," ",Gammas_all[j])
                into = Gammas_all[i+1]*rhos[i+1, j+1]
                outof = Gammas_all[i]*rhos[i, j]
                decay_matrix[i,j] = into - outof
            
            else
                outof = Gammas_all[i]*rhos[i, j]
                decay_matrix[i,j] = - outof
                
            end
           
        end
    end
      
    return np.matrix(decay_matrix)
    
end

function L_dephasing(gammas)
    """
    Given list of laser linewidths, create dephasing operator in matrix form. 

    Args:
        gammas (list): Linewidths of the field coupling each pair of states 
            in the ladder, from lowest to highest energy states 
            (gamma_{12}, ..., gamma{n-1,n} for n levels).

    Returns:
        numpy.matrix: Dephasing operator as a matrix populated by expressions 
            containing Sympy symbolic objects.
            Will be size nxn for an n-level system.
    """    
    levels = length(gammas)+1
    rhos = def_rho_matrix(levels)
    deph_matrix = np.zeros((levels, levels), dtype = "object")
    for i in 1:levels
        for j in 1:levels
            if i != j
                max=np.max([i,j])
                #if max == levels
                 #   max = levels-1
                #else max = np.max([i,j])
                #end
                deph_matrix[i,j] = -(np.sum(gammas[np.min(
                    [i,j]):max-1]))*rhos[i,j]
            end
        end
    end        
    return np.matrix(deph_matrix)
end

function Master_eqn(H_tot, L)
    """
    Return an expression for the right hand side of the Master equation 
    (as in Eqn 18) for a given Hamiltonian and Lindblad term. 
    Assumes that the number of atomic levels is set by the shape of the 
    Hamiltonian matrix (nxn for n levels).

    Args:
        H_tot (matrix): The total Hamiltonian in the interaction picture.
        L (matrix): The Lindblad superoperator in matrix form. 

    Returns:
        numpy.matrix: Right hand side of the Master equation (eqn 18) 
        as a matrix containing expressions in terms of Sympy symbolic objects.
        Will be size nxn for an n-level system.
    """    
    levels = size(H_tot,1)
    dens_mat = def_rho_matrix(levels)
    return -1im*(H_tot*dens_mat - dens_mat*H_tot) + L
end
Deltas = [1,2]
Omegas = [2,3]
Gammas = [5,1]
gammas = [0.1,0.2]

#x= def_rho_matrix(3)
Ldecay = L_decay(Gammas)
Ldeph = L_dephasing(gammas)
L = Ldecay+Ldeph
H = Hamiltonian(Omegas, Deltas)
x=Master_eqn(H,L)
println(x)
