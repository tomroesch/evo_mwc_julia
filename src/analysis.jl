function get_growth_rate(file, meta_data_file, stan_file)
    df = read_OD(file, meta_data_file)
    
    s = open(stan_file) do file
        read(file, String)
    end
    growth = Array{Float64, 1}[]
    rate = Array{Float64, 1}[]
    bw_model = Stanmodel(
      name="bw", 
      nchains=4,
      num_warmup=1000,
      num_samples=1000,
      thin=10,
      model=s,
      printsummary=false
    )
    max_rates = zeros(Float64, 48)
    for (j, well) in enumerate(unique(df[!, :well]))
        t = df[df.well .== well, Symbol("time_[s]")]
        N = df[df.well .== well, :OD]
        strain = df[df.well .== well, :strain] |> unique
        if strain[1] != "blank"
            bw_data = Dict(    
                "N"=>length(t),
                "x"=>t,
                "y"=>N,
                "N_predict"=>72,
                "x_predict"=>range(minimum(t), stop=maximum(t), length=72),
            )


            _, bw_chains, bw_names = stan(bw_model, bw_data, summary=false)

            d = [collect_params_from_chain(bw_names, bw_chains[:,:,i]) for i in 1:4]
            chain_maximums = map(x->maximum(mean(x["g_predict"], dims=1)), d)
            max_rates[j] = maximum(chain_maximums)
            println(string("Well ", well, " done"))

            chain_means_growth = map(x->mean(x["y_predict"], dims=1),d)
            mean_growths = mean.([[chain_means_growth[i][j] for i in 1:4] for j in 1:72])
            chain_means_rates = map(x->mean(x["g_predict"], dims=1),d)
            mean_rates = mean.([[chain_means_rates[i][j] for i in 1:4] for j in 1:72])

            push!(growth, mean_growths)
            push!(rate, mean_rates)
            
        end
    end
    return max_rates, growth, rate
end
        
            
            