from src import analysis


def main():
    rk = analysis.conduct_rank_correlation(
        n=500,
        epochs=30,
        device="cuda"
    )
    rk.write_to_csv()
    rk.print()


if __name__ == '__main__':
    main()