import { useEffect, useState } from 'react';
import NativeZedLLM from '../../../craby-modules/src/NativeZedLLM';
import type { LanguageModelInfo, AgentServerInfo } from '../types';
import { normalizeAgentServerInfo, normalizeLanguageModelInfo } from '../utils/nativeJson';

export interface AvailableBackends {
  models: LanguageModelInfo[];
  agents: AgentServerInfo[];
  defaultModelId: string | null;
}

export function useAvailableBackends(): AvailableBackends {
  const [models, setModels] = useState<LanguageModelInfo[]>([]);
  const [agents, setAgents] = useState<AgentServerInfo[]>([]);
  const [defaultModelId, setDefaultModelId] = useState<string | null>(null);

  useEffect(() => {
    const refresh = () => {
      try {
        const modelsJson = NativeZedLLM.getModels();
        const parsedModels: unknown[] = JSON.parse(modelsJson);
        setModels(parsedModels.map(normalizeLanguageModelInfo));

        const defaultId = NativeZedLLM.getDefaultModelId();
        setDefaultModelId(defaultId);

        const agentsJson = NativeZedLLM.getAgentServers();
        const parsedAgents: unknown[] = JSON.parse(agentsJson);
        setAgents(parsedAgents.map(normalizeAgentServerInfo));
      } catch (error) {
        console.error('Failed to load backends:', error);
      }
    };

    refresh();

    NativeZedLLM.addListener('agentConnected');
    NativeZedLLM.addListener('agentDisconnected');
    const unsubConnected = NativeZedLLM.agentConnected(() => refresh());
    const unsubDisconnected = NativeZedLLM.agentDisconnected(() => refresh());

    return () => {
      unsubConnected();
      unsubDisconnected();
      NativeZedLLM.removeListeners(2);
    };
  }, []);

  return { models, agents, defaultModelId };
}
