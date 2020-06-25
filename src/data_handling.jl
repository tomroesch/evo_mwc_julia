function read_OD(file, meta_data_file)
    # Read file per line
    s = open(file) do f
        readlines(f)
    end
    # Look for OD header
    OD_header = findall(x-> occursin("OD", x), s)[1]
    
    # Find empty lines
    empty_lines = findall(x->x=="", s)
    
    # Read OD data
    if length(empty_lines)==1
        df = CSV.read(file, skipto=OD_header+2)
    else
        ind = findfirst(x->x>OD_header, empty_lines)
        df = CSV.read(file, datarow=OD_header+2, limit=empty_lines[ind+1]-empty_lines[ind]-1, header=false)
    end
    # Rename time and temperature
    rename!(df, Dict("Column1"=>"time_[s]", "Column2"=>"temp_[C]"))
    # Make tidy dataframe
    df  = stack(df, 3:ncol(df))
    
    # Rename columns
    rename!(df, Dict("variable"=>"Column", "value"=>"OD"))
    
    # Drop missing data
    dropmissing!(df)
    
    # Transform time to seconds from start
    df[!, Symbol("time_[s]")] = map(x->Dates.value(Dates.Second(x - Dates.Time(0))), df[!, Symbol("time_[s]")])
    
    xf = XLSX.readxlsx(meta_data_file)
    meta_df = DataFrame()
    for x in XLSX.sheetnames(xf)
        arr = convert(Array{String},xf[x]["A1:H6"])
        flat_arr = vcat([arr[i, :] for i in 1:size(arr, 1)]...)
        meta_df[!, Symbol(x)]= flat_arr
    end
    meta_df[! , :Column] = [Symbol("Column$i") for i in 3:50]
    return join(meta_df, df, on = :Column, kind=:outer)
end
