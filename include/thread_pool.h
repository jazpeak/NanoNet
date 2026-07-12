#pragma once

#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <stdexcept>
using namespace std;

class ThreadPool {
public:
    ThreadPool(size_t threads);
    ~ThreadPool();
    void enqueue(function<void()> task);

private:
    vector<thread> workers;
    queue<function<void()>> tasks;
    mutex queue_mutex;
    condition_variable condition;
    bool stop;
};

ThreadPool& get_thread_pool();
