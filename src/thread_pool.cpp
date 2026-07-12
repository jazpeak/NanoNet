#include "thread_pool.h"

ThreadPool::ThreadPool(size_t threads) {
    stop = false;
    for (size_t i = 0; i < threads; ++i) {
        workers.push_back(thread([this]() {
            while (true) {
                function<void()> task;

                unique_lock<mutex> lock(queue_mutex);
                condition.wait(lock, [this]() {
                    return stop == true || tasks.empty() == false;
                });

                if (stop == true && tasks.empty() == true) {
                    return;
                }

                task = tasks.front();
                tasks.pop();
                
                lock.unlock();

                task();
            }
        }));
    }
}

ThreadPool::~ThreadPool() {
    unique_lock<mutex> lock(queue_mutex);
    stop = true;
    lock.unlock();

    condition.notify_all();

    for (thread &worker : workers) {
        worker.join();
    }
}

void ThreadPool::enqueue(function<void()> task) {
    unique_lock<mutex> lock(queue_mutex);
    if (stop == true) {
        throw runtime_error("enqueue on stopped ThreadPool");
    }
    tasks.push(task);
    lock.unlock();

    condition.notify_one();
}

ThreadPool& get_thread_pool() {
    static ThreadPool pool(thread::hardware_concurrency());
    return pool;
}
