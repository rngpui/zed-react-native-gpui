// ============================================================================
// Language Model Types (not parsed by craby)
// ============================================================================

export interface LanguageModelInfo {
  id: string;
  name: string;
  providerId: string;
  providerName: string;
  supportsImages: boolean;
  supportsTools: boolean;
}

export interface LanguageModelProviderInfo {
  id: string;
  name: string;
  isAuthenticated: boolean;
}

// ============================================================================
// Agent Server Types
// ============================================================================

export type AgentServerType = 'codex' | 'claude' | 'gemini' | 'custom';

export interface AgentServerInfo {
  type: AgentServerType;
  name: string;
  isConnected: boolean;
  customName?: string;
}

// ============================================================================
// Tool Call Types
// ============================================================================

export type ToolKind = 'read' | 'write' | 'execute' | 'network' | 'other';

export type ToolCallStatus =
  | 'pending'
  | 'waiting_for_confirmation'
  | 'in_progress'
  | 'completed'
  | 'failed'
  | 'rejected'
  | 'canceled';

export interface ToolCallInfo {
  id: string;
  requestId?: number;
  name: string;
  title: string;
  kind: ToolKind;
  status: ToolCallStatus;
  inputJson?: string;
  resultText?: string;
  errorMessage?: string;
  /** Number of diffs available for this tool call (for write operations). */
  diffCount: number;
  /** Number of terminals available for this tool call (for execute operations). */
  terminalCount: number;
}

export interface PermissionOption {
  id: string;
  label: string;
  description?: string;
  isDefault: boolean;
}

export interface ToolAuthorizationRequest {
  toolCallId: string;
  title: string;
  options: PermissionOption[];
}

export type ToolCallEvent =
  | { type: 'started'; toolCall: ToolCallInfo }
  | { type: 'status_changed'; id: string; status: ToolCallStatus }
  | { type: 'result_ready'; id: string; resultText: string }
  | { type: 'authorization_required'; request: ToolAuthorizationRequest }
  | { type: 'error'; id: string; error: string }
  | { type: 'content_updated'; id: string; diffCount: number; terminalCount: number };
