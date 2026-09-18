from src.logging.logger_interface import Logger


class SupernetLogger(Logger):

    def log(
            self,
            epoch: int,
            total_epochs: int,
            train_acc: float,
            val_accuracies: list[float]
    ) -> None:
        self.buffer.append([
            f"Epoch {epoch + 1}/{total_epochs}:",
            f"Training Accuracy: {train_acc:.2f} %",
            ] + [f"Val Accuracy {n}: {val_acc:.2f}" for n, val_acc in enumerate(val_accuracies)]
        )

        if self._buffer_is_full() or self._training_is_complete(epoch, total_epochs):
            self._print_buffer()

    @staticmethod
    def _training_is_complete(epoch: int, total_epochs: int):
        return (epoch + 1) == total_epochs
