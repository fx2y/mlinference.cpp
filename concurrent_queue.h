#include <queue>
#include <mutex>
#include <condition_variable>

// To implement request pipelining, we can use a producer-consumer pattern with a thread-safe queue:
// The ConcurrentQueue class provides thread-safe push, pop, and empty operations using a mutex and a condition variable.
template <typename T>
class ConcurrentQueue
{
public:
    void push(const T &item)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        queue_.push(item);
        lock.unlock();
        condVar_.notify_one();
    }

    T pop()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        condVar_.wait(lock, [this]
                      { return !queue_.empty(); });
        T item = queue_.front();
        queue_.pop();
        return item;
    }

    bool empty()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.empty();
    }

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable condVar_;
};
