import multiprocessing.pool
import threading
import asyncio

async def process_with_progress_bar(*, items, process_fn, progress_bar_description):
    from tqdm.asyncio import tqdm_asyncio
    return await tqdm_asyncio.gather(*[process_fn(item) for item in items], desc=progress_bar_description)

def join_threads():
    for thread in threading.enumerate():
        if thread.daemon:
            continue

        try:
            thread.join()
        except RuntimeError as error:
            if 'cannot join current thread' in error.args[0]: # main thread
                pass
            else:
                raise
