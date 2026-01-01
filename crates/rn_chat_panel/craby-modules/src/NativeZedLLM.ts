import type { NativeModule, Signal } from '@rngpui/craby-modules';
import { NativeModuleRegistry } from '@rngpui/craby-modules';

// Re-export complex types from non-craby file
export type {
  LanguageModelInfo,
  LanguageModelProviderInfo,
  AgentServerType,
  AgentServerInfo,
  ToolKind,
  ToolCallStatus,
  ToolCallInfo,
  PermissionOption,
  ToolAuthorizationRequest,
  ToolCallEvent,
} from '../LLMTypes';

// ============================================================================
// Signal Event Types (craby-compatible, no unions or optional properties)
// ============================================================================

export interface LLMChunkEvent {
  requestId: number;
  chunk: string;
}

export interface LLMDoneEvent {
  requestId: number;
}

export interface LLMErrorEvent {
  requestId: number;
  error: string;
}

export interface AgentConnectedEvent {
  agentType: string; // AgentServerType as string for craby
  sessionId: string;
}

export interface AgentDisconnectedEvent {
  agentType: string; // AgentServerType as string for craby
  reason: string;
}

export interface ToolCallEventEvent {
  event: string; // JSON-encoded ToolCallEvent
}

export interface ThreadSnapshotPayload {
  snapshot: string; // JSON-encoded ThreadSnapshot
}

export interface AcpThreadsChangedEvent {
  threads: string; // JSON-encoded AcpThreadMetadata[]
}

// ============================================================================
// TurboModule Spec
// ============================================================================

export interface Spec extends NativeModule {
  // Language Model Methods
  isAvailable(): boolean;
  getProviders(): string; // JSON-encoded LanguageModelProviderInfo[]
  getModels(): string; // JSON-encoded LanguageModelInfo[]
  getDefaultModelId(): string; // empty string if none

  // Send message to default model (backwards compatible)
  sendMessage(message: string): number;
  // Send message to a specific model
  sendMessageToModel(modelId: string, message: string): number;
  cancelRequest(requestId: number): void;

  // Agent Server Methods
  getAgentServers(): string; // JSON-encoded AgentServerInfo[]
  connectAgent(agentType: string): boolean;
  disconnectAgent(agentType: string): void;
  sendMessageToAgent(agentType: string, message: string): number;

  // Agent configuration (per-agent, best-effort; JSON-encoded for flexibility)
  ensureAgentThread(agentType: string): boolean;
  startAgentThread(agentType: string): boolean;
  getAgentModels(agentType: string): string; // JSON-encoded AgentModelInfo[]
  setAgentModel(agentType: string, modelId: string): boolean;
  getAgentModes(agentType: string): string; // JSON-encoded AgentModeInfo[]
  setAgentMode(agentType: string, modeId: string): boolean;

  // Signals for language model responses
  llmChunk: Signal<LLMChunkEvent>;
  llmDone: Signal<LLMDoneEvent>;
  llmError: Signal<LLMErrorEvent>;

  // Signals for agent server events
  agentConnected: Signal<AgentConnectedEvent>;
  agentDisconnected: Signal<AgentDisconnectedEvent>;

  // Tool Call Methods
  respondToToolAuthorization(toolCallId: string, permissionOptionId: string): boolean;
  cancelToolCall(toolCallId: string): boolean;

  // Signal for tool call events (event is JSON-encoded ToolCallEvent)
  toolCallEvent: Signal<ToolCallEventEvent>;

  // Signal for thread snapshots (payload is JSON-encoded ThreadSnapshot)
  threadSnapshot: Signal<ThreadSnapshotPayload>;

  // Signal for ACP thread history updates (payload is JSON-encoded)
  acpThreadsChanged: Signal<AcpThreadsChangedEvent>;

  // ACP thread history
  listAcpThreads(): string; // JSON-encoded AcpThreadMetadata[]
  searchAcpThreads(query: string): string; // JSON-encoded AcpThreadMetadata[]
  openAcpThread(threadId: string): boolean;

  // Chat pane controls
  setChatPaneVisible(visible: boolean): void;
  minimizeChatPane(): void;

  // Required by NativeEventEmitter
  addListener(eventName: string): void;
  removeListeners(count: number): void;
}

export default NativeModuleRegistry.getEnforcing<Spec>('ZedLLM');
