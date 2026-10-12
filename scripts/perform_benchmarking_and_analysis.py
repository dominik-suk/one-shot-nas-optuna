import matplotlib as mpl

from src import analysis


def analyze_spos(random_search: bool = False):
    analysis.benchmark_spos(random_search)

    if not random_search:
        analysis.plot_spos_nsga_versus_random()

    analysis.plot_spos_heatmap(random_search)
    analysis.plot_spos_learning_curve(random_search)
    analysis.plot_spos_bar_chart(
        random_search=random_search,
        do_save=True,
    )


def analyze_baseline():
    analysis.benchmark_baseline()
    analysis.plot_baseline_heatmap()
    analysis.plot_baseline_learning_curve()
    analysis.plot_baseline_bar_chart(
        do_save=True,
    )


def main():
    mpl.use("pgf")
    mpl.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    })

    analyze_spos()
    analyze_spos(random_search=True)
    analyze_baseline()


if __name__ == "__main__":
    analysis.benchmark_constraint_spos()
    # main()
