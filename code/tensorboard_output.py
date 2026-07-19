"""TensorBoard scalar writer without a TensorFlow dependency."""

import time

import numpy as np
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter


class TensorBoardOutput:
    """Write scalar logger summaries to a standard TensorBoard event file."""

    def __init__(self, logdir, _fps=20):
        self._writer = EventFileWriter(str(logdir))

    def __call__(self, summaries):
        wall_time = time.time()
        for step, name, value in summaries:
            value = np.asarray(value)
            if value.shape or not np.issubdtype(value.dtype, np.number):
                continue
            summary = Summary(value=[
                Summary.Value(tag=name, simple_value=float(value)),
            ])
            self._writer.add_event(Event(
                wall_time=wall_time, step=int(step), summary=summary))
        self._writer.flush()

    def wait(self):
        self._writer.flush()
