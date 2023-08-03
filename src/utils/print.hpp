#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

#ifndef NDEBUG
#  define MIMI_FUNC() (std::cout << "*** calling (" << __func__ << ") ***\n";)
#else
#  define MIMI_F()
#endif

namespace mimi::utils {

template<typename... Args>
void PrintI(Args&&... args) {
  std::cout << "MIMI INFO - ";
  ((std::cout << std::forward<Args>(args) << " "), ...);
  std::cout << "\n";
}

/*
 * debug printer - first argument is bool,
 * so <on, off> is switchable
 */
template<typename... Args>
void PrintD(Args&&... args) {
#ifndef NDEBUG
    std::cout << "MIMI DEBUG - ";
    ((std::cout << std::forward<Args>(args) << " "), ...);
    std::cout << "\n";
#endif
}

template<typename... Args>
void PrintW(Args&&... args) {
  std::cout << "MIMI WARNING - ";
  ((std::cout << std::forward<Args>(args) << " "), ...);
  std::cout << "\n";
}

template<typename... Args>
void PrintE(Args&&... args) {
  std::stringstream msg{};
  msg << "MIMI ERROR - ";
  ((msg << std::forward<Args>(args) << " "), ...);
  msg << "\n";
  throw std::runtime_error(msg.str());
}

}