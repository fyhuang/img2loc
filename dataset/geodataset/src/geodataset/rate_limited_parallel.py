import concurrent.futures
import threading
import queue

# These two are only for DaemonThreadPoolExecutor
import concurrent.futures.thread
import weakref

class _DaemonThreadPoolExecutor(concurrent.futures.ThreadPoolExecutor):
    """A ThreadPoolExecutor that doesn't block program exit.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _adjust_thread_count(self):
        # if idle threads are available, don't spin new threads
        if self._idle_semaphore.acquire(timeout=0):
            return

        # When the executor gets lost, the weakref callback will wake up
        # the worker threads.
        def weakref_cb(_, q=self._work_queue):
            q.put(None)

        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread_name = '%s_%d' % (self._thread_name_prefix or self,
                                     num_threads)
            t = threading.Thread(name=thread_name, target=concurrent.futures.thread._worker,
                                 args=(weakref.ref(self, weakref_cb),
                                       self._work_queue,
                                       self._initializer,
                                       self._initargs),
                                 # NEW: set daemon to not block program exit
                                 daemon=True)
            t.start()
            self._threads.add(t)
            #concurrent.futures.thread._threads_queues[t] = self._work_queue

    def shutdown(self, wait=True, *, cancel_futures=False):
        super().shutdown(wait=False, cancel_futures=cancel_futures)
        if wait:
            # Wait but with timeout
            for t in self._threads:
                t.join(timeout=0.5)


class RateLimitedExecutor:
    def __init__(self, rate_limit, max_workers=8):
        self.rate_limit = rate_limit
        self.semaphore = threading.Semaphore(1)
        self.quit_queue = queue.Queue()
        self.filler_thread = threading.Thread(target=self._token_filler)

        self.executor = _DaemonThreadPoolExecutor(max_workers=max_workers)

    def _token_filler(self):
        secs_per_token = 1.0 / self.rate_limit
        while True:
            try:
                self.quit_queue.get(timeout=secs_per_token)
                print("Got the exit signal")
                return
            except queue.Empty:
                self.semaphore.release()

    def _rate_limit_wrapper(self, fn, *args, **kwargs):
        self.semaphore.acquire()
        return fn(*args, **kwargs)

    def submit(self, fn, *args, **kwargs):
        if not self.filler_thread.is_alive():
            self.filler_thread.start()

        return self.executor.submit(self._rate_limit_wrapper, fn, *args, **kwargs)

    def shutdown(self):
        self.quit_queue.put(True)
        if self.filler_thread.is_alive():
            self.filler_thread.join(timeout=10)
        self.executor.shutdown(wait=True, cancel_futures=True)
