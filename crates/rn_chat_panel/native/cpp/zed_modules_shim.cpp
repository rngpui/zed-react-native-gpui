// SPDX-License-Identifier: GPL-3.0-or-later
//
// C++ shim for registering Zed's TurboModules with RNGPUI.
// This file provides the callback function that RNGPUI calls during
// TurboModule registry creation.

#include "CrabyzedmodulesRegistration.hpp"

#include <ReactCommon/CallInvoker.h>
#include <ReactCommon/TurboModule.h>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

// Mirror of gpui::modules::TurboModuleEntry from RNGPUI's ModulesRegistry.h
// We define it locally to avoid complex include path dependencies.
namespace gpui::modules {
struct TurboModuleEntry {
  bool devOnly{false};
  std::function<std::shared_ptr<facebook::react::TurboModule>()> factory;
};
using TurboModuleRegistry = std::unordered_map<std::string, TurboModuleEntry>;
}  // namespace gpui::modules

extern "C" void zed_register_turbomodules(void* registry_ptr, void* invoker_ptr) {
  auto& registry = *static_cast<gpui::modules::TurboModuleRegistry*>(registry_ptr);

  // Create a non-owning shared_ptr from the raw CallInvoker pointer.
  // RNGPUI owns the CallInvoker; we just borrow it during registration.
  auto invoker = std::shared_ptr<facebook::react::CallInvoker>(
      static_cast<facebook::react::CallInvoker*>(invoker_ptr),
      [](auto*) {}  // no-op deleter (non-owning)
  );

  // Register ZedTheme, ZedLLM, ZedWorkspace turbomodules
  craby::zedmodules::registration::registerModules(registry, invoker);
}
