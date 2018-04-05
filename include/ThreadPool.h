//
// Created by yang chen on 2018/3/12.
// Modified based on ThreadPool(https://github.com/progschj/ThreadPool).
//

#ifndef MINICNN_THREADPOOL_H
#define MINICNN_THREADPOOL_H

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>


namespace MiniCNN
{
    class ThreadPool
    {
    public:
        static ThreadPool& instance();
        unsigned int size() const;
        void resize(const unsigned int size);
        template<class F, class... Args>
        auto enqueue(F&& f, Args&&... args)
        ->std::future<typename std::result_of<F(Args...)>::type>;

    private:
        ThreadPool(const unsigned int threads);
        virtual ~ThreadPool();
        void shutdown();
        void startup(const unsigned int threads);

    private:
        std::vector<std::thread> m_workers;
        std::queue<std::function<void()>> m_tasks;

        std::mutex m_queueMutex;
        std::condition_variable m_condition;
        bool m_stop = true;
    };

    template<class F, class...Args>
    auto ThreadPool::enqueue(F &&f, Args &&... args)->std::future<typename std::result_of<F(Args...)>::type>
    {
        using return_tyep = typename std::result_of<F(Args...)>::type;
        auto task = std::make_shared<std::packaged_task<return_tyep()>>
                (std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        std::future<return_tyep> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(m_queueMutex);

            if (m_stop)
                throw std::runtime_error("enqueue on stopped ThreadPool");

            m_tasks.emplace([task](){(*task)();});
        }

        m_condition.notify_one();
        return res;
    }


    unsigned int get_thread_num();
    unsigned int set_thread_num(const unsigned int num);
    void dispatch_worker(std::function<void(const unsigned int, const unsigned int)> func, const unsigned int number);
}



#endif //MINICNN_THREADPOOL_H