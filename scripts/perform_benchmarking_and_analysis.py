from src import analysis


def analyze_spos(random_search: bool = False):
    analysis.benchmark_spos(random_search)
    rk = analysis.conduct_rank_correlation(n=25)
    rk.write_to_csv()
    rk.print()

    analysis.plot_spos_heatmap(random_search)
    analysis.plot_spos_learning_curve(random_search)
    analysis.plot_spos_nsga_versus_random()
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
    analyze_spos()
    analyze_spos(random_search=True)
    analyze_baseline()


if __name__ == "__main__":
    main()
