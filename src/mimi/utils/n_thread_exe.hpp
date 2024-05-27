#pragma once

#include <algorithm>

#ifdef MIMI_USE_OMP
#include <omp.h>
#elif MIMI_USE_BS_POOL
#include "mimi/utils/BS_thread_pool.hpp"
#endif

#include <thread>

#include "mimi/utils/print.hpp"

namespace mimi::utils {

/// @brief helper function that returns id of
/// @param i_thread
/// @return
inline int ThisThreadId(const int i_thread) {
#ifdef MIMI_USE_BS_POOL
  // return static_cast<int>(*BS::this_thread::get_index());
  return i_thread; // static_cast<int>(*BS::this_thread::get_index());
#elif MIMI_USE_OMP
  return static_cast<int>(omp_get_thread_num());
#else
  // this has no meaningful statement
  return i_thread;
#endif
}

#ifdef MIMI_USE_BS_POOL
extern BS::thread_pool thread_pool;
#endif

/// @brief multi thread execution helper based on chunked batches
/// @tparam Func
/// @tparam IndexT
/// @param f expected to have the following signature
///          -> f(const int begin, const int end, const int i_thread)
///          It is your choice to use which variable
/// @param total
/// @param nthread
template<typename Func, typename IndexT>
void NThreadExe(const Func& f, const IndexT total, const IndexT nthread) {
  // if nthread == 1, don't even bother creating thread
  if (nthread == 1 || nthread == 0) {
    f(0, total, 0);
    return;
  }

  IndexT n_usable_threads{nthread};

  // negative input looks for hardware_concurrency
  if (nthread < 0) {
    n_usable_threads = std::max(std::thread::hardware_concurrency(), 1u);
  }

  // thread shouldn't exceed total
  n_usable_threads = std::min(total, n_usable_threads);

  // get chunk size and prepare threads
  const IndexT chunk_size = (total + n_usable_threads - 1) / n_usable_threads;

#ifdef MIMI_USE_OMP
  auto from = [&](const int i) {
    if (i < n_usable_threads - 1) {
      return i * chunk_size;
    } else {
      return (n_usable_threads - 1) * chunk_size;
    }
  };
  auto to = [&](const int i) {
    if (i < n_usable_threads - 1) {
      return (i + 1) * chunk_size;
    } else {
      return total;
    }
  };

#pragma omp parallel for
  for (int i = 0; i < n_usable_threads; ++i) {
    f(from(i), to(i), omp_get_thread_num());
  }

#elif MIMI_USE_BS_POOL
  // need to pass n usable threads, else it may give max threads
  thread_pool.detach_blocks<int>(0, total, f, n_usable_threads);
  thread_pool.wait();
#else
  std::vector<std::thread> tpool;
  tpool.reserve(n_usable_threads);

  for (int i{0}; i < (n_usable_threads - 1); i++) {
    tpool.emplace_back(std::thread{f, i * chunk_size, (i + 1) * chunk_size, i});
  }
  {
    // last one
    tpool.emplace_back(std::thread{f,
                                   (n_usable_threads - 1) * chunk_size,
                                   total,
                                   n_usable_threads - 1});
  }

  for (auto& t : tpool) {
    t.join();
  }
#endif
}

} // namespace mimi::utils
