include("Julia_OBE_tools-2-level-trial.jl")
using Plots
using FileIO

Delta_0 = 5.266e-8 #eV
Omegas = [1.398e9, 1.398e9]
Gammas = [6.25e-11, 6.25e-11]
Deltas = [Gammas[1]/2, Gammas[2]/2]

# Define initial density matrix
rho_0 = reshape([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (9, 1))
rho_t = zeros(Float64, length(times), 4)

#Initial conditions atom 
T = 2.0e-3 #K
k_b = 1.38e-23 #J/K
m = 1.443e-25 #kg
x_position = 0.0
y_position = 0.0
x_velocity = sqrt(k_b*T/m)
y_velocity = 0.0
println("x_velocity ", x_velocity)
#time
dt = 0.1e-9
time = 32.0e-9

#Constants
g = 1.0
mu_B = 8.794e10 #1/sT

k = 1.06e7
B = 1.0

#Force 


# Define arrays to store position, velocity, time and force
x_positions = Float64[]
y_positions = Float64[]
x_velocities = Float64[]
time_list = Float64[]
force_list = Float64[]
for t in 1:Int(time / dt)
    push!(x_positions, x_position)
    push!(y_positions, y_position)
    push!(x_velocities, x_velocity)
    push!(time_list, t*dt)
    
    update_detuning(Deltas,Delta_0,g,mu_B,B,k,x_velocity) # Update detunings
    
    M = time_dep_matrix(Omegas, Deltas, Gammas,k,x_position) # Calculate the time dependent matrix
    
    density_array_t = time_evolve(M, t, rho_0) # Perform time evolution of the density matrix
    density_mat_t = reshape(density_array_t, (3, 3)) # Reshape the density matrix into a 3x3 matrix
    F_0 = force_operator(Omegas,k,x_position)
    F_t = real(expected_value(F_0, density_mat_t))# Calculate the expected value of the force operator
    
    global x_position, y_position, x_velocity, y_velocity = update_velocity_position(x_position,y_position,x_velocity,y_velocity,dt,F_t)
    
end

p = plot(time_list, x_positions, xlabel="time", ylabel="x position", title="atom_in_mot")
savefig(p, "atom_in_mot.png")
p = plot(time_list, x_velocities, xlabel="time", ylabel="x velocity",  title="atom_in_mot velocity")
savefig(p, "atom_in_mot_velocity.png")
# Plot the trajectory of the ball
println("x_position ", x_positions)
anim = @animate for i in 1:length(x_positions)
    plot([x_positions[i]], [y_positions[i]], seriestype=:scatter, marker=:circle, ms=10,
        xlim=(-10e-9, 10e-9), ylim=(-2e-9, 15e-9), xlabel="X Position", ylabel="Y Position",
        title="atom_in_mot", legend=false)
end

gif(anim, "atom_in_mot.gif", fps = 20)