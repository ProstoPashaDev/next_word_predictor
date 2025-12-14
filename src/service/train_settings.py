from tensorflow.keras.callbacks import Callback


class StopOnLossThreshold(Callback):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get("loss")
        if loss is not None and loss <= self.threshold:
            print(f"\nStopping training: loss {loss:.4f} <= {self.threshold}")
            self.model.stop_training = True
