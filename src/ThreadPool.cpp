//
// Created by yang chen on 2018/3/12.
//

#include "../include/ThreadPool.h"
#include <algorithm>

namespace MiniCNN
{
    ThreadPool& ThreadPool::instance()
    {
        static ThreadPool inst(2);
        return inst;
    }
    // the constructor just launches some amount of workers
    ThreadPool::ThreadPool(const unsigned int threads)
    {
        startup(threads);
    }

    // the destructor joins all threads
    ThreadPool::~ThreadPool()
    {
        shutdown();
    }

    void ThreadPool::shutdown()
    {
        {
            std::unique_lock<std::mutex> lock(m_queueMutex);
            m_stop = true;
        }

        m_condition.notify_all();
        for (std::thread &worker : m_workers)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }

        m_workers.clear();
        m_tasks = decltype(m_tasks)();
    }

    void ThreadPool::startup(const unsigned int threads)
    {
        m_stop = false;
        for (unsigned int i = 0; i < threads; ++i)
            m_workers.emplace_back(
                [this]
                {
                    for (;;)
                    {
                        std::function<void()> task;

                        {
                            std::unique_lock<std::mutex> lock(this->m_queueMutex);
                            this->m_condition.wait(lock, [this]{ return this->m_stop || !this->m_tasks.empty(); });
                            if (this->m_stop && this->m_tasks.empty())
                                break;
                            task = std::move(this->m_tasks.front());
                            this->m_tasks.pop();
                        }

                        task();
                    }
                }
            );
    }
    inline unsigned int ThreadPool::size() const
    {
        return m_workers.size();
    }
    void ThreadPool::resize(const unsigned int new_size)
    {
        //stop thread pool
        shutdown();
        //start thread pool
        startup(new_size);
    }

    unsigned int get_thread_num()
    {
        return ThreadPool::instance().size();
    }

    unsigned int set_thread_num(const unsigned int num)
    {
        if (num != get_thread_num())
        {
            ThreadPool::instance().resize(num);
        }
        return get_thread_num();
    }

    void dispatch_worker(std::function<void(const unsigned int, const unsigned int)> func, const unsigned int number)
    {
        if (number <= 0)
        {
            return;
        }

        const unsigned int threads_of_pool = ThreadPool::instance().size();

        if (threads_of_pool <= 1 || number <= 1)
        {
            func(0, number);
        }
        else
        {
            // 1/4 2/4 /4/4 5/4 => all ok!
            const unsigned int payload_per_thread = number / threads_of_pool;
            const unsigned int remainder_payload = number - payload_per_thread*threads_of_pool;
            const unsigned int remainder_proc_last_idx = remainder_payload;

            unsigned int start = 0;
            std::vector<std::future<void>> futures;
            for (unsigned int i = 0; i < threads_of_pool; i++)
            {
                unsigned int stop = start + payload_per_thread;
                if (i < remainder_proc_last_idx)
                {
                    stop = stop + 1;
                }
                futures.push_back(ThreadPool::instance().enqueue(func, start, stop));
                start = stop;
                if (stop >= number)
                {
                    break;
                }
            }
            for (unsigned int i = 0; i < futures.size(); i++)
            {
                futures[i].wait();
            }
        }
    }
}//namespace
