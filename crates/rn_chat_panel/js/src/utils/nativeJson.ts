import type {
  AgentServerInfo,
  LanguageModelInfo,
  ToolAuthorizationRequest,
  ToolCallEvent,
  ToolCallInfo,
} from '../types';

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

export function normalizeLanguageModelInfo(raw: unknown): LanguageModelInfo {
  if (!isObject(raw)) throw new Error('Invalid LanguageModelInfo');
  return {
    id: String(raw.id ?? ''),
    name: String(raw.name ?? ''),
    providerId: String(raw.providerId ?? raw.provider_id ?? ''),
    providerName: String(raw.providerName ?? raw.provider_name ?? ''),
    supportsImages: Boolean(raw.supportsImages ?? raw.supports_images ?? false),
    supportsTools: Boolean(raw.supportsTools ?? raw.supports_tools ?? false),
  };
}

export function normalizeAgentServerInfo(raw: unknown): AgentServerInfo {
  if (!isObject(raw)) throw new Error('Invalid AgentServerInfo');
  return {
    type: String(raw.type ?? '') as AgentServerInfo['type'],
    name: String(raw.name ?? ''),
    isConnected: Boolean(raw.isConnected ?? raw.is_connected ?? false),
    customName: (raw.customName ?? raw.custom_name) ? String(raw.customName ?? raw.custom_name) : undefined,
  };
}

export function normalizeToolCallInfo(raw: unknown): ToolCallInfo {
  if (!isObject(raw)) throw new Error('Invalid ToolCallInfo');
  const requestIdRaw = raw.requestId ?? raw.request_id;
  return {
    id: String(raw.id ?? ''),
    requestId: typeof requestIdRaw === 'number' ? requestIdRaw : requestIdRaw ? Number(requestIdRaw) : undefined,
    name: String(raw.name ?? ''),
    title: String(raw.title ?? ''),
    kind: String(raw.kind ?? 'other') as ToolCallInfo['kind'],
    status: String(raw.status ?? 'pending') as ToolCallInfo['status'],
    inputJson: raw.inputJson ?? raw.input_json ? String(raw.inputJson ?? raw.input_json) : undefined,
    resultText: raw.resultText ?? raw.result_text ? String(raw.resultText ?? raw.result_text) : undefined,
    errorMessage: raw.errorMessage ?? raw.error_message ? String(raw.errorMessage ?? raw.error_message) : undefined,
    diffCount: Number(raw.diffCount ?? raw.diff_count ?? 0),
    terminalCount: Number(raw.terminalCount ?? raw.terminal_count ?? 0),
  };
}

export function normalizeToolAuthorizationRequest(raw: unknown): ToolAuthorizationRequest {
  if (!isObject(raw)) throw new Error('Invalid ToolAuthorizationRequest');
  const optionsRaw = Array.isArray(raw.options) ? raw.options : [];
  return {
    toolCallId: String(raw.toolCallId ?? raw.tool_call_id ?? ''),
    title: String(raw.title ?? ''),
    options: optionsRaw.map((opt) => {
      if (!isObject(opt)) throw new Error('Invalid PermissionOption');
      return {
        id: String(opt.id ?? ''),
        label: String(opt.label ?? ''),
        description: opt.description ? String(opt.description) : undefined,
        isDefault: Boolean(opt.isDefault ?? opt.is_default ?? false),
      };
    }),
  };
}

export function normalizeToolCallEvent(raw: unknown): ToolCallEvent {
  if (!isObject(raw)) throw new Error('Invalid ToolCallEvent');
  const type = String(raw.type ?? '');
  switch (type) {
    case 'started':
      return {
        type: 'started',
        toolCall: normalizeToolCallInfo(raw.toolCall ?? raw.tool_call),
      };
    case 'status_changed':
      return {
        type: 'status_changed',
        id: String(raw.id ?? ''),
        status: String(raw.status ?? 'pending') as any,
      };
    case 'result_ready':
      return {
        type: 'result_ready',
        id: String(raw.id ?? ''),
        resultText: String(raw.resultText ?? raw.result_text ?? ''),
      };
    case 'authorization_required':
      return {
        type: 'authorization_required',
        request: normalizeToolAuthorizationRequest(raw.request),
      };
    case 'error':
      return {
        type: 'error',
        id: String(raw.id ?? ''),
        error: String(raw.error ?? ''),
      };
    case 'content_updated':
      return {
        type: 'content_updated',
        id: String(raw.id ?? ''),
        diffCount: Number(raw.diffCount ?? raw.diff_count ?? 0),
        terminalCount: Number(raw.terminalCount ?? raw.terminal_count ?? 0),
      };
    default:
      throw new Error(`Unknown ToolCallEvent type: ${type}`);
  }
}
