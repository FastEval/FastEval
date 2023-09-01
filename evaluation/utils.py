import multiprocessing.pool
import threading
import asyncio

async def process_with_progress_bar_async(items, process_fn, progress_bar_description):
    from tqdm.asyncio import tqdm_asyncio

    stop_event = threading.Event()

    return await tqdm_asyncio.gather(
        *[process_fn(item, stop_event=stop_event) for item in items],
        desc=progress_bar_description
    )

def process_with_progress_bar(*, items, process_fn, progress_bar_description):
    return asyncio.run(process_with_progress_bar_async(items, process_fn, progress_bar_description))


    async def process_with_index(item_with_index):
        index, item = item_with_index
        result = await process_fn(item, stop_event=stop_event)
        return index, result

    try:
        with multiprocessing.pool.ThreadPool(min(num_threads, len(items))) as pool:
            iterator = pool.imap_unordered(process_with_index, enumerate(items))
            results_with_indices = list(tqdm.tqdm(iterator, total=len(items), desc=progress_bar_description))
    except Exception as exception:
        if stop_event is not None:
            stop_event.set()
        raise exception

    return [result_with_index[1] for result_with_index in sorted(results_with_indices, key=lambda item: item[0])]

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
