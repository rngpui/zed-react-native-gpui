export type {
  LanguageModelInfo,
  AgentServerInfo,
  ToolKind,
  ToolCallStatus,
  ToolCallInfo,
  PermissionOption,
  ToolAuthorizationRequest,
  ToolCallEvent,
} from '../../../craby-modules/src/NativeZedLLM';

export type Model = import('../../../craby-modules/src/NativeZedLLM').LanguageModelInfo;
export type Agent = import('../../../craby-modules/src/NativeZedLLM').AgentServerInfo;

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  streaming?: boolean;
  chunks?: import('../hooks/useThreadSnapshot').AssistantChunkSnapshot[];
}

export interface AcpThreadMetadata {
  id: string;
  title: string;
  updatedAt: string;
}

export type AgentAccessModeId = 'read_only' | 'agent' | 'agent_full_access';

export interface AgentModelInfo {
  id: string;
  name: string;
  description?: string | null;
  selected: boolean;
}

export interface AgentModeInfo {
  id: string;
  name: string;
  description?: string | null;
  selected: boolean;
}

export interface WorkspaceInfo {
  projectName: string | null;
  rootPath: string | null;
  currentFilePath: string | null;
}
