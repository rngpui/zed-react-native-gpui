#pragma once

#include "rust/cxx.h"
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace craby {
namespace zedmodules {
namespace bridging {
  struct ZedLLMSignal;
}
namespace modules {
  class CxxZedLlm;
}
}
}

namespace craby {
namespace zedmodules {
namespace signals {

using Delegate = std::function<void(const std::string& signalName, void* signal)>;

class SignalManager {
public:
  static SignalManager& getInstance() {
    static SignalManager instance;
    return instance;
  }

  // Takes uintptr_t signal pointer from Rust (type-erased since different modules have different signal types)
void emit(uintptr_t id, rust::Str name, size_t signal) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = delegates_.find(id);
    if (it != delegates_.end()) {
      it->second(std::string(name), reinterpret_cast<void*>(signal));
    }
  }

  void registerDelegate(uintptr_t id, Delegate delegate) const {
    std::lock_guard<std::mutex> lock(mutex_);
    delegates_.insert_or_assign(id, delegate);
  }

  void unregisterDelegate(uintptr_t id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    delegates_.erase(id);
  }

private:
  SignalManager() = default;
  mutable std::unordered_map<uintptr_t, Delegate> delegates_;
  mutable std::mutex mutex_;
};

inline const SignalManager& getSignalManager() {
  return SignalManager::getInstance();
}

} // namespace signals
} // namespace zedmodules
} // namespace craby
