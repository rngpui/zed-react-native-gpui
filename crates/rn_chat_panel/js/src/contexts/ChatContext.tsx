import React, {
  createContext,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from 'react';

import NativeZedLLM from '../../../craby-modules/src/NativeZedLLM';
import type { Agent, AgentAccessModeId, Message } from '../types';
import { useThreadSnapshot, type ThreadSnapshot } from '../hooks/useThreadSnapshot';
import { messagesFromSnapshot } from '../utils/threadSnapshot';

interface ChatContextType {
  messages: Message[];
  inputValue: string;
  setInputValue: (value: string) => void;
  isLoading: boolean;

  selectedAgent: Agent | null;
  setSelectedAgent: (agent: Agent | null) => void;

  agentAccessMode: AgentAccessModeId;
  setAgentAccessMode: (mode: AgentAccessModeId) => void;

  isFollowing: boolean;
  setIsFollowing: (following: boolean) => void;

  currentThreadId: string | null;
  currentThreadTitle: string;
  threadSnapshot: ThreadSnapshot | null;

  sendMessage: () => void;
  startNewThread: (agentType?: string) => boolean;
}

const ChatContext = createContext<ChatContextType | null>(null);

function buildThreadTitle(messages: Message[]): string {
  const firstUserMsg = messages.find((m) => m.role === 'user');
  if (firstUserMsg) {
    const content = firstUserMsg.content.trim();
    return content.length > 50 ? content.substring(0, 47) + '...' : content;
  }
  return 'Thread';
}

interface ChatProviderProps {
  children: ReactNode;
  sessionId: string;
}

export function ChatProvider({ children }: ChatProviderProps) {
  const [inputValue, setInputValue] = useState('');
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [agentAccessMode, setAgentAccessMode] = useState<AgentAccessModeId>('agent');
  const [isFollowing, setIsFollowing] = useState<boolean>(false);
  const threadSnapshot = useThreadSnapshot();
  const messages = useMemo(() => messagesFromSnapshot(threadSnapshot), [threadSnapshot]);
  const isLoading = threadSnapshot?.status === 'generating';
  const currentThreadId = threadSnapshot?.session_id ?? null;
  const currentThreadTitle =
    threadSnapshot?.title && threadSnapshot.title.length > 0
      ? threadSnapshot.title
      : buildThreadTitle(messages);

  const sendMessage = () => {
    const text = inputValue.trim();
    if (!text || isLoading) return;

    if (!selectedAgent) {
      console.warn('No agent selected.');
      return;
    }

    setInputValue('');

    const connected = NativeZedLLM.connectAgent(selectedAgent.type);
    if (!connected) {
      setSelectedAgent(null);
      console.warn('Failed to connect the agent.');
      return;
    }

    NativeZedLLM.ensureAgentThread(selectedAgent.type);
    NativeZedLLM.sendMessageToAgent(selectedAgent.type, text);
  };

  const startNewThread = (agentType?: string): boolean => {
    const resolvedAgent = agentType ?? selectedAgent?.type;
    if (!resolvedAgent) {
      console.warn('No agent selected for new thread.');
      return false;
    }

    const connected = NativeZedLLM.connectAgent(resolvedAgent);
    if (!connected) {
      console.warn('Failed to connect the agent.');
      return false;
    }

    if (!NativeZedLLM.startAgentThread(resolvedAgent)) {
      console.warn('Failed to start a new thread.');
      return false;
    }
    return true;
  };

  const value: ChatContextType = {
    messages,
    inputValue,
    setInputValue,
    isLoading,
    selectedAgent,
    setSelectedAgent,
    agentAccessMode,
    setAgentAccessMode,
    isFollowing,
    setIsFollowing,
    currentThreadId,
    currentThreadTitle,
    threadSnapshot,
    sendMessage,
    startNewThread,
  };

  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}

export function useChat(): ChatContextType {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
}
