import type { AssistantChunkSnapshot, EntrySnapshot, ThreadSnapshot } from '../hooks/useThreadSnapshot';
import type { Message, ToolAuthorizationRequest, ToolCallInfo } from '../types';

type ToolCallEntry = Extract<EntrySnapshot, { type: 'tool_call' }>;

function toSnakeCase(value: string): string {
  return value
    .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
    .replace(/[\s-]+/g, '_')
    .toLowerCase();
}

function normalizeToolCallKind(kind: string): ToolCallInfo['kind'] {
  const normalized = toSnakeCase(kind);
  if (normalized.includes('read')) return 'read';
  if (normalized.includes('write') || normalized.includes('edit')) return 'write';
  if (normalized.includes('execute') || normalized.includes('terminal') || normalized.includes('run')) return 'execute';
  if (normalized.includes('network') || normalized.includes('fetch') || normalized.includes('http') || normalized.includes('web')) {
    return 'network';
  }
  return 'other';
}

function normalizeToolCallStatus(status: string): ToolCallInfo['status'] {
  const normalized = toSnakeCase(status);
  switch (normalized) {
    case 'pending':
    case 'waiting_for_confirmation':
    case 'in_progress':
    case 'completed':
    case 'failed':
    case 'rejected':
    case 'canceled':
      return normalized as ToolCallInfo['status'];
    default:
      return 'pending';
  }
}

function assistantChunksToText(chunks: AssistantChunkSnapshot[]): string {
  const textChunks = chunks.filter((chunk) => chunk.type === 'text');
  if (textChunks.length === 0) {
    return '';
  }
  return textChunks.map((chunk) => chunk.content).join('');
}

export function messagesFromSnapshot(snapshot: ThreadSnapshot | null): Message[] {
  if (!snapshot) return [];

  const { entries } = snapshot;
  const lastAssistantIndex = entries.reduce(
    (last, entry, index) => (entry.type === 'assistant_message' ? index : last),
    -1
  );
  const isGenerating = snapshot.status === 'generating';

  const messages: Message[] = [];
  entries.forEach((entry, index) => {
    if (entry.type === 'user_message') {
      messages.push({
        id: entry.id ?? `user-${index}`,
        role: 'user',
        content: entry.content,
      });
      return;
    }

    if (entry.type === 'assistant_message') {
      messages.push({
        id: `assistant-${index}`,
        role: 'assistant',
        content: assistantChunksToText(entry.chunks),
        chunks: entry.chunks,
        streaming: isGenerating && index === lastAssistantIndex,
      });
    }
  });

  return messages;
}

export function toolCallFromEntry(entry: ToolCallEntry): ToolCallInfo {
  const name = entry.kind ? toSnakeCase(entry.kind) : entry.title;

  return {
    id: entry.id,
    name,
    title: entry.title,
    kind: normalizeToolCallKind(entry.kind),
    status: normalizeToolCallStatus(entry.status),
    inputJson: entry.input_json ?? undefined,
    resultText: entry.result_text ?? undefined,
    errorMessage: entry.error_message ?? undefined,
    diffCount: entry.diff_count,
    terminalCount: entry.terminal_count,
  };
}

export function toolCallsFromSnapshot(snapshot: ThreadSnapshot | null): ToolCallInfo[] {
  if (!snapshot) return [];
  const toolCalls: ToolCallInfo[] = [];
  for (const entry of snapshot.entries) {
    if (entry.type === 'tool_call') {
      toolCalls.push(toolCallFromEntry(entry));
    }
  }
  return toolCalls;
}

export function authorizationRequestsFromSnapshot(
  snapshot: ThreadSnapshot | null
): ToolAuthorizationRequest[] {
  if (!snapshot) return [];
  const requests: ToolAuthorizationRequest[] = [];
  for (const entry of snapshot.entries) {
    if (entry.type !== 'tool_call') continue;
    const options = entry.authorization_options;
    if (!options || options.length === 0) continue;
    requests.push({
      toolCallId: entry.id,
      title: entry.title,
      options: options.map((option) => ({
        id: option.id,
        label: option.label,
        description: option.description ?? undefined,
        isDefault: option.is_default,
      })),
    });
  }
  return requests;
}
