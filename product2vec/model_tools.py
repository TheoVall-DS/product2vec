"""This module provides extended epoch logger for Product2Vec model.

"""
from collections import deque
from time import perf_counter

from gensim.models.callbacks import CallbackAny2Vec


class EpochLogger(CallbackAny2Vec):
    """EpochLogger logs training progress for gensim model.

    Parameters
    ----------
    n_latest : int, default=5
        Estimate time left for training with this number of latest epoch durations.

    References
    -----
    Original implementation: https://radimrehurek.com/gensim/models/callbacks.html
    """

    def __init__(self, n_latest: int = 5):
        self._start_stamp = 0
        self._current_epoch = 1
        self._epoch_duration = deque(maxlen=n_latest)

    @staticmethod
    def format_time(seconds: int) -> str:
        """Convert time in seconds to a readable time format."""
        formatted_time = ""

        hours = seconds // 3600
        if hours > 0:
            hours_as_str = str(hours) if len(str(hours)) > 1 else "0" + str(hours)
            formatted_time += hours_as_str + ":"

        minutes = (seconds % 3600) // 60
        minutes_as_str = str(minutes) if len(str(minutes)) > 1 else "0" + str(minutes)
        formatted_time += minutes_as_str + ":"

        seconds_left = seconds - hours * 3600 - minutes * 60
        seconds_as_str = (
            str(seconds_left) if len(str(seconds_left)) > 1 else "0" + str(seconds_left)
        )
        formatted_time += seconds_as_str

        return formatted_time

    def on_epoch_begin(self, model) -> None:
        """Print current epoch and estimated time left."""
        self._start_stamp = perf_counter()

        if self._epoch_duration:
            avg_duration = sum(self._epoch_duration) / len(self._epoch_duration)
            time_left = (model.epochs - (self._current_epoch - 1)) * avg_duration
            msg = self.format_time(int(time_left))
        else:
            msg = "To be estimated"

        print(f"Epoch #{self._current_epoch}. Estimated time left - {msg}")

    def on_epoch_end(self, model) -> None:
        """Update epoch number and duration."""
        duration = perf_counter() - self._start_stamp
        self._epoch_duration.append(duration)
        self._current_epoch += 1
