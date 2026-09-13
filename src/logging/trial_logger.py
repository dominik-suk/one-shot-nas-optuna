import optuna

from src.logging.logger_interface import Logger


class TrialLogger(Logger):
    def __init__(self, n_trials: int, buffer_size: int = 5):
        super().__init__(buffer_size=buffer_size)
        self.n_trials = n_trials

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        self.log(study=study, trial=trial)

    def log(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        self.buffer.append([
            f"Trial {trial.number + 1}/{self.n_trials}:",
            f"Accuracy: {trial.value:.2f} %",
            f"Best Trial: Nr. {study.best_trial.number + 1}",
            f"Best Accuracy: {study.best_trial.value:.2f} %",
        ])

        if self._buffer_is_full() or self._optimization_is_complete(trial=trial):
            self._print_buffer()

    def _optimization_is_complete(self, trial: optuna.trial.FrozenTrial) -> bool:
        return trial.number + 1 == self.n_trials
