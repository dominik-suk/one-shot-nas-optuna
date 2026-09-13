from src.logging.logger_interface import Logger


class TrainLogger(Logger):

    def log(
            self,
            epoch: int,
            total_epochs: int,
            train_acc: float,
            val_acc: float,
            current_best_acc: float,
            val_loss: float,
            f1_score: float,
    ) -> None:
        self.buffer.append([
            f"Epoch {epoch + 1}/{total_epochs}:",
            f"Validation Loss: {val_loss:.4f}",
            f"Training Accuracy: {train_acc:.2f} %",
            f"Validation Accuracy: {val_acc:.2f} %",
            f"Best Accuracy: {current_best_acc:.2f} %",
            f"F1 Score: {f1_score:.2f} %",
        ])

        if self._buffer_is_full() or self._training_is_complete(epoch, total_epochs):
            self._print_buffer()

    @staticmethod
    def _training_is_complete(epoch: int, total_epochs: int):
        return (epoch + 1) == total_epochs
