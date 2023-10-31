using GenericLinearAlgebra
using SymPy
using PyCall
using PyPlot
np = pyimport("numpy")

function create_rho_list(levels = 3)
    rho_list = Sym[]
    for i in 1:levels
        for j in 1:levels
            push!(rho_list,Sym("rho_$(i)$(j)"))
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

function OBE_matrix(Master_matrix)
    """
    Take the right hand side of the Master equation (-i[H,\rho] + L) 
    expressed as an array of multiples of Sympy symbolic objects and output 
    an ndarray of coefficients M such that d rho_vect/dt = M*rho_vect 
    where rho_vect is the vector of density matrix elements.

    Args:
        Master_matrix (matrix): The right hand side of the Master equation 
            as a matrix of multiples of Sympy symbolic objects

    Returns:
        numpy.ndarray: An array of (complex) coefficients. 
            Will be size n^2 x n^2 for an n-level system.
    """    
    levels = size(Master_matrix,1)
    rho_vector = create_rho_list(levels)
    coeff_matrix = np.zeros((levels^2, levels^2), dtype = "complex")
    count = 1
    for i in 1:levels
        for j in 1:levels
            entry = Master_matrix[i,j]
            expanded = sympy.expand(entry)
            #use Sympy coeff to extract coefficient of each element in rho_vect
            for (n,r) in enumerate(rho_vector)
                coeff_matrix[count, n] = complex(expanded.coeff(sympy.sympify(r)))
            end 
            count += 1
        end
    end    
    return coeff_matrix
end

function SVD(coeff_matrix)
    """
    Perform singular value decomposition (SVD) on matrix of coefficients 
    using the numpy.linalg.svd function.
    SVD returns US(V)*^T where S is the diagonal matrix of singular values.
    The solution of the system of equations is then the column of V 
    corresponding to the zero singular value.
    If there is no zero singular value (within tolerance, allowing for 
    floating point precision) then return a zero array.

    Args:
        coeff_matrix (ndarray): array of complex coefficients M that satisfies 
            the expression d rho_vect/dt = M*rho_vect where rho_vect is the 
            vector of density matrix elements.

    Returns:
        ndarray: 1D array of complex floats corresponding to the steady-state 
            value of each element of the density matrix in the order given by 
            the rho_list function. Will be length n for an n-level system.
    """    
    levels = Int(np.sqrt(size(coeff_matrix,1)))
    u,sig,v = np.linalg.svd(coeff_matrix)
    abs_sig = np.abs(sig)
    minval = np.min(abs_sig)
    if minval>1e-12
        println("ERROR - Matrix is non-singular")
        return np.zeros((levels^2))
    end
    index = argmin(abs_sig)[1]
    rho = np.conjugate(v[index,:]) 
    #SVD returns the conjugate transpose of v
    pops = np.zeros((levels))
    for l in 0:levels-1
        pops[l+1] = (np.real(rho[l*(levels+1)+1]))
    end
    t = 1/(np.sum(pops)) #Normalise so the sum of the populations is one
    rho_norm = rho*t
    return rho_norm
end

function steady_state_soln(Omegas, Deltas, Gammas, gammas = [])
    """
    Given lists of parameters (all in order of lowers to highest energy level), 
    construct and solve for steady state of density matrix. 
    Returns ALL elements of density matrix in order of rho_list 
    (rho_11, rho_12, ... rho_i1, rho_i2, ..., rho_{n, n-1}, rho_nn).

    Args:
        Omegas (list of floats): Rabi frequencies of fields coupling each pair 
            of states in the ladder, from lowest to highest energy states 
            (Omega_{12}, ..., Omega{n-1,n} for n levels).
        Deltas (list of floats): Detuning of fields coupling each pair 
            of states in the ladder, from lowest to highest energy states 
            (Delta_{12}, ..., Delta{n-1,n} for n levels).
        Gammas (list of floats): Linewidth of the atomic states considered. 
            Assumes no decay from lowest energy atomic state (Gamma_1 = 0).
            Values should be given in order of lowest to highest energy level 
            (Gamma_2, Gamma_3, ... , Gamma_n for n levels). 
            n-1 values for n levels.
        gammas (list of floats, optional): Linewidths of the fields coupling 
            each pair of states in the ladder, from lowest to highest energy 
            (gamma_{12}, ..., gamma{n-1,n} for n levels). 
            Defaults to [] (meaning all gamma_ij = 0).

    Returns:
        ndarray: 1D array containing values for each element of the density 
            matrix in the steady state, in the order returned by the rho_list 
            function (rho_11, rho_12, ..., rho_{n, n-1}, rho_nn).
            Will be length n for an n-level system.
    """    
    L_atom = L_decay(Gammas)
    if length(gammas) != 0
        L_laser = L_dephasing(gammas)
        L_tot = L_atom + L_laser
    
    else
        L_tot = L_atom
    end
    H = Hamiltonian(Omegas, Deltas)
    Master = Master_eqn(H, L_tot)
    rho_coeffs = OBE_matrix(Master)
    soln = SVD(rho_coeffs)
    return soln
end


Deltas = [1]
Omegas = [2]
Gammas = [5]
gammas = [0.1]

#x= def_rho_matrix(3)
Ldecay = L_decay(Gammas)
Ldeph = L_dephasing(gammas)
L = Ldecay+Ldeph
H = Hamiltonian(Omegas, Deltas)
master=Master_eqn(H,L)
coeff_mat=OBE_matrix(master)
svd=SVD(coeff_mat)
x = steady_state_soln(Omegas, Deltas, Gammas, gammas)


Omegas = [1,5] #2pi MHz
Deltas = [0.0,0.0] #2pi MHz
Gammas = [0.5,0.1] #2pi MHz
# Create symmetric array of values for probe detuning (\Delta_{12})
Delta_12 = np.linspace(-10,10,200) #2pi MHz
probe_abs = np.zeros((2, length(Delta_12))) #Create blank array to store solution

for (i, p) in enumerate(Delta_12)
    Deltas[1] = p # update value of \Delta_{12}
    solution = steady_state_soln(Omegas, Deltas, Gammas)
    probe_abs[1,i] = imag(solution[2])
    # repeat but with the second field 'turned off' (\Omega_{23} = 0)
    solution = steady_state_soln([1,0], Deltas, Gammas)
    probe_abs[2,i] = imag(solution[2])
end

figure()
plot(Delta_12, probe_abs[2,:], label = "\\Omega_{23} = 0 MHz")
plot(Delta_12, probe_abs[1,:], label = "\\Omega_{23} = 5 MHz")
xlabel("Probe detuning \\Delta_{12} (MHz)")
ylabel("Probe absorption (-\\Im[\\rho_{21}])")
legend(loc = "best")
show()