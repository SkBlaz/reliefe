if Base.find_package("Relief") === nothing
    println(stderr, "Install Julia package Relief 0.2.0")
    exit(42)
end
using Relief
using Random

folder, method, width, seed, feature_type = ARGS[1:5]
seed == "None" || Random.seed!(parse(Int, seed))
p = parse(Int, width)
data = permutedims(reshape(ltoh.(reinterpret(UInt64, read(joinpath(folder, "features")))), p, :))
x = copy(reinterpret(Float64, data))
y = Int.(ltoh.(reinterpret(Int64, read(joinpath(folder, "labels")))))
functions = Dict("reliefmss" => Relief.reliefmss, "boostedsurf" => Relief.boostedsurf,
                 "iterative-relief" => Relief.iterative_relief, "irelief" => Relief.irelief,
                 "ec-relieff" => Relief.ecrelieff)
if method == "irelief"
    weights = Relief.irelief(x, y, parse(Int, ARGS[6]), parse.(Float64, ARGS[7:end])...; f_type=feature_type)
else
    weights = functions[method](x, y, parse.(Int, ARGS[6:end])...; f_type=feature_type)
end
write(joinpath(folder, "scores"), htol.(reinterpret(UInt64, Float64.(vec(weights)))))
