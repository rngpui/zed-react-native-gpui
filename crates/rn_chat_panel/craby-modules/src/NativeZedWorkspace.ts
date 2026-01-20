import type { NativeModule, Signal } from '@rngpui/craby-modules';
import { NativeModuleRegistry } from '@rngpui/craby-modules';

export interface WorkspaceInfo {
  projectName: string | null;
  rootPath: string | null;
  currentFilePath: string | null;
}

export interface WorkspaceChangedEvent {
  workspace: WorkspaceInfo;
}

export interface Spec extends NativeModule {
  getWorkspaceInfo(): WorkspaceInfo;
  workspaceChanged: Signal<WorkspaceChangedEvent>;
  // Required by NativeEventEmitter - no-ops since events go through __rctDeviceEventEmitter
  addListener(eventName: string): void;
  removeListeners(count: number): void;
}

export default NativeModuleRegistry.getEnforcing<Spec>('ZedWorkspace');
