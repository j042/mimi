#pragma once

#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>

#ifndef NDEBUG
#define MIMI_FUNC()                                                            \
  (std::cout << "\n☎️ (" << __PRETTY_FUNCTION__ << " - " << __FILE__ << ":"     \
             << __LINE__ << ") \n");
#else
#define MIMI_FUNC()
#endif

namespace mimi::utils {

static std::mutex print_mutex;

template<typename... Args>
void PrintInfo(Args&&... args) {
  std::cout << "MIMI INFO - ";
  ((std::cout << std::forward<Args>(args) << " "), ...);
  std::cout << "\n";
}

/*
 * debug printer - first argument is bool,
 * so <on, off> is switchable
 */
template<typename... Args>
void PrintDebug(Args&&... args) {
#ifndef NDEBUG
  std::cout << "MIMI DEBUG - ";
  ((std::cout << std::forward<Args>(args) << " "), ...);
  std::cout << "\n";
#endif
}

template<typename... Args>
void PrintWarning(Args&&... args) {
  std::cout << "MIMI WARNING - ";
  ((std::cout << std::forward<Args>(args) << " "), ...);
  std::cout << "\n";
}

template<typename... Args>
void PrintAndThrowError(Args&&... args) {
  std::stringstream msg{};
  msg << "MIMI ERROR - ";
  ((msg << std::forward<Args>(args) << " "), ...);
  msg << "\n";
  throw std::runtime_error(msg.str());
}

template<typename... Args>
void PrintSynced(Args&&... args) {
  std::lock_guard<std::mutex> guard(print_mutex);

  std::cout << "MIMI SYNCED - ";
  ((std::cout << std::forward<Args>(args) << " "), ...);
  // std::cout << "\n";
  std::cout << std::endl;
  ;
}

} // namespace mimi::utils
