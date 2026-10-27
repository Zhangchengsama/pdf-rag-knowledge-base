"""为原处理步骤记录耗时，通过可选回调报告状态。"""
from contextlib import contextmanager
from time import perf_counter


@contextmanager
def track_stage(stage, callback=None):
    started = perf_counter()
    def emit(status):
        if callback is not None:
            callback({'stage': stage, 'status': status,
                      'elapsed_seconds': perf_counter() - started})
    emit('running')
    try:
        yield
    except Exception:
        emit('failed')
        raise
    else:
        emit('complete')
