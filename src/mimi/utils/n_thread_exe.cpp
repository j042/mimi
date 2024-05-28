#include "mimi/utils/n_thread_exe.hpp"
#include "mimi/utils/BS_thread_pool.hpp"
namespace mimi::utils {

#ifdef MIMI_USE_BS_POOL
BS::thread_pool thread_pool;
#else
std::array<Worker, 24> workers;
#endif
} // namespace mimi::utils
