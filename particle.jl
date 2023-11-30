include("Julia_OBE_tools-2-level-trial.jl")
using Plots
using FileIO

Delta_0 = -8.44e-27 #J
Omegas = [1.398e9, 1.398e9]
Gammas = [6.25e-11, 6.25e-11]
Deltas = [-Gammas[1]/2, -Gammas[2]/2]

# Define initial density matrix
rho_0 = reshape([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (9, 1))
rho_t = zeros(Float64, length(times), 4)

#Initial conditions atom 
T = 2.0e-3 #K
k_b = 1.38e-23 #J/K
m = 3.82e-26 #kg
x_position = 0.0 #m
y_position = 0.0 #m
x_velocity = sqrt(k_b*T/m) #m/s
y_velocity = 0.0 #m/s
#println("x_velocity ", x_velocity)
#time
dt = 0.2e-9 #s
time = 150.0e-9 #s

#Constants
g = 1.0 
mu_B = 8.794e10 #1/sT

k = 1.06e7 #m^-1 
B = 1e-7 #T

#Force 


# Define arrays to store position, velocity, time and force
x_positions = Float64[]
y_positions = Float64[]
x_velocities = Float64[]
time_list = Float64[]
force_list = ComplexF64[]

# Perform time evolution
for t in 1:Int(round(time / dt))
    push!(x_positions, x_position)
    push!(y_positions, y_position)
    push!(x_velocities, x_velocity)
    push!(time_list, t*dt)
    
    #update_detuning(Deltas,Delta_0,g,mu_B,B,k,x_velocity) # Update detunings
    update_detuning2(Deltas,Delta_0,g,mu_B,B,k,x_velocity,x_position) #!Variable magnetic field
    M = time_dep_matrix(Omegas, Deltas, Gammas,k,x_position) # Calculate the time dependent matrix
    
    density_array_t = time_evolve(M, t, rho_0) # Perform time evolution of the density matrix
    density_mat_t = reshape(density_array_t, (3, 3)) # Reshape the density matrix into a 3x3 matrix
    F_0 = force_operator(Omegas,k,x_position)
    push!(force_list, expected_value(F_0, density_mat_t))
    F_t = real(expected_value(F_0, density_mat_t))# Calculate the expected value of the force operator
    
    global x_position, y_position, x_velocity, y_velocity = update_velocity_position(x_position,y_position,x_velocity,y_velocity,dt,F_t)
    
end

p = plot(time_list, x_positions, xlabel="time(s)", ylabel="x position(m)", title="atom_in_mot")
savefig(p, "atom_in_mot.png")
p = plot(time_list, x_velocities, xlabel="time(s)", ylabel="x velocity (m)",  title="atom_in_mot velocity")
savefig(p, "atom_in_mot_velocity.png")
# Plot the trajectory of the ball
open("not_im_force.txt", "w") do f
   # println(f,"force_list ", force_list)
end
anim = @animate for i in 1:length(x_positions)
    plot([x_positions[i]], [y_positions[i]], seriestype=:scatter, marker=:circle, ms=10,
        xlim=(-2e-6, 2e-6), ylim=(-2e-6, 2e-6), xlabel="X Position(m)", ylabel="Y Position(m)",
        title="atom_in_mot", legend=false)
end

gif(anim, "atom_in_mot.gif", fps = 20)