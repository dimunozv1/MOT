using PyCall
using Plots
using FileIO
# Constants
G = 6.674 * (10^-11)  # gravitational constant
mass_of_earth = 5.972 * (10^24)  # mass of Earth in kg


function calculate_force(mass1, mass2, distance)
    return mass2*9.8
end

function update_velocity_position(x,y,vx,vy,dt)
    # assuming the earth at 0,0 the origin
    r = sqrt(x^2 + y^2)
    force = calculate_gravity(mass_of_earth, 1, r)

    theta = atan(y/x)
    fx = 0
    fy = -force 

    #update vel and pos 
    vx += fx * dt
    vy += fy * dt
    x += vx * dt
    y += vy * dt

    return x,y,vx,vy

end

#Initial conditions
x_position = 0.0
y_position = 10.0
x_velocity = 0.0
y_velocity = 10.0

dt = 0.1
time = 3

x_positions = Float64[]  # Use Float64[] instead of []
y_positions = Float64[]  # Use Float64[] instead of []
time_list = Float64[]    # Use Float64[] instead of []

for t in 1:Int(time / dt)
    push!(x_positions, x_position)
    push!(y_positions, y_position)
    push!(time_list, t*dt)
    
   global x_position, y_position, x_velocity, y_velocity = update_velocity_position(x_position, y_position, x_velocity, y_velocity, dt)
end
println("x: ", x_positions, " y: ", y_positions)
p = plot(time_list, y_positions)

savefig(p, "ball_movement.png")
# Plot the trajectory of the ball

anim = @animate for i in 1:length(x_positions)
    plot([x_positions[i]], [y_positions[i]], seriestype=:scatter, marker=:circle, ms=10,
        xlim=(-15, 15), ylim=(-2, 15), xlabel="X Position", ylabel="Y Position",
        title="Ball Movement Under Gravity", legend=false)
end

gif(anim, "ball_movement.gif", fps = 20)
