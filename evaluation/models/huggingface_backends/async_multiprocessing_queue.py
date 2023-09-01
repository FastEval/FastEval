class AsyncMultiprocessingQueue:
    """
    multiprocessing.Queue
    - Doesn't work with asyncio because it always blocks
    - Needs to be created in the parent process.
      It's not possible to create it in the child process and then send it to the parent.

    asyncio.Queue
    - Only works with asyncio and not with multiprocessing.

    We need a queue that can
    - Work with multiprocessing and can also be created in the child processes
    - Work with asyncio

    Apparently there is no built-in implementation for this, so we have to do it ourself...
    """

    def __init__(self):
        raise Exception('TODO')

    def put(self, item):
        raise Exception('TODO')

    async def get(self):
        raise Exception('TODO')
