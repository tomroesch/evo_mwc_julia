function read_OD(file)
    s = open(file) do f
        readlines(f)
    end
    OD_header = findall(x-> occursin("OD", x), s)[1]
    empty_lines = findall(x->x=="", s)
    
    if length(empty_lines)==1
        df = CSV.read(file, skipto=OD_header+2)
    else
        ind = findfirst(x->x>OD_header, empty_lines)
        df = CSV.read(file, datarow=OD_header+2, limit=empty_lines[ind+1]-empty_lines[ind]-1, header=false)
    end
    rename!(df, Dict("Column1"=>"Time", "Column2"=>"Temperature"))
    df  = stack(df, 3:ncol(df))
    rename!(df, Dict("variable"=>"Well", "value"=>"OD"))
    return df
end
