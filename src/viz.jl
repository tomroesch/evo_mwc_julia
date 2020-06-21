using Plots

function default_plotlyjs!()
    plotlyjs(
        background_color="#E3DCD0",
        foreground_color_grid="#ffffff",
        gridlinewidth= 0.5,
        guidefontfamily="Lucida Sans Unicode",
        guidefontsize=8,
        tickfontfamily="Lucida Sans Unicode",
        tickfontsize=7,
        titlefontfamily="Lucida Sans Unicode",
        titlefontsize=8,
        dpi=300,
        linewidth=1.25,
        legendtitlefontsize=8,
        legendfontsize=8,
        foreground_color_legend="#E3DCD0",
        color_palette=:seaborn_colorblind
    )
end