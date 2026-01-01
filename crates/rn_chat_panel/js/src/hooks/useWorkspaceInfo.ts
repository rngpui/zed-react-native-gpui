import { useState, useEffect } from 'react';
import type { WorkspaceInfo } from '../types';

export function useWorkspaceInfo() {
  const [info, setInfo] = useState<WorkspaceInfo | null>(null);

  useEffect(() => {
    try {
      const { NativeModuleRegistry } = require('@rngpui/craby-modules');
      const mod = NativeModuleRegistry.get('ZedWorkspace');
      if (mod?.getWorkspaceInfo) {
        const data = mod.getWorkspaceInfo();
        setInfo(data);

        if (mod.workspaceChanged) {
          mod.addListener('workspaceChanged');
          const unsubscribe = mod.workspaceChanged((event: { workspace: WorkspaceInfo }) => {
            setInfo(event.workspace);
          });
          return () => {
            unsubscribe();
            mod.removeListeners(1);
          };
        }
      }
    } catch (error) {
      console.error('Failed to get workspace info:', error);
    }
  }, []);

  return info;
}
