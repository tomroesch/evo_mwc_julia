using Plots


"""
    default_plotlyjs!()

Set plotting default to that used in Physical Biology of the Cell, 2nd edition.
"""
function default_plotlyjs!()
    plotlyjs(
        background_color="#E3DCD0",
        background_color_outside="white",
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