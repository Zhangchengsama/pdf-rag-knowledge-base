"""为原处理步骤记录耗时，通过可选回调报告状态。"""
from contextlib import contextmanager
from time import perf_counter


@contextmanager
def track_stage(stage, callback=None):
    # 阶段计时
    # 1. 记录开始时间，并通知阶段开始。
    # 2. 执行原步骤，分别报告完成或失败。
    # 3. 失败时继续抛出原异常。
    started = perf_counter()
    def emit(status):
        # 报告阶段
        # 1. 有回调时发送阶段名称、状态和已用时间。
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
