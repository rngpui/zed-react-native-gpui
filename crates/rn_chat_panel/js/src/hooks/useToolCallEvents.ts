import { useEffect, useMemo, useRef } from 'react';
import NativeZedLLM from '../../../craby-modules/src/NativeZedLLM';
import { useChat } from '../contexts';
import { authorizationRequestsFromSnapshot, toolCallsFromSnapshot } from '../utils/threadSnapshot';
import type { ToolAuthorizationRequest, ToolCallInfo } from '../types';

function pickOption(
  request: ToolAuthorizationRequest,
  kind: 'allow' | 'deny',
): string | null {
  const lowerIncludes = (text: string | undefined, query: string) =>
    typeof text === 'string' && text.toLowerCase().includes(query);

  const options = request.options ?? [];

  if (kind === 'deny') {
    const deny =
      options.find((o) => lowerIncludes(o.label, 'deny')) ??
      options.find((o) => lowerIncludes(o.label, 'reject')) ??
      options.find((o) => lowerIncludes(o.label, 'cancel'));
    return deny?.id ?? null;
  }

  const allow =
    options.find((o) => lowerIncludes(o.label, 'always')) ??
    options.find((o) => lowerIncludes(o.label, 'allow')) ??
    options.find((o) => o.isDefault);
  return allow?.id ?? null;
}

export function useToolCallEvents() {
  const { threadSnapshot, agentAccessMode } = useChat();

  const toolCalls = useMemo(() => toolCallsFromSnapshot(threadSnapshot), [threadSnapshot]);
  const pendingAuthorizations = useMemo(
    () => authorizationRequestsFromSnapshot(threadSnapshot),
    [threadSnapshot],
  );

  const handledAuthRequestsRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    if (agentAccessMode === 'agent') return;

    const kind = agentAccessMode === 'read_only' ? 'deny' : 'allow';

    for (const request of pendingAuthorizations) {
      if (handledAuthRequestsRef.current.has(request.toolCallId)) continue;
      const optionId = pickOption(request, kind);
      if (!optionId) continue;

      const ok = NativeZedLLM.respondToToolAuthorization(request.toolCallId, optionId);
      if (ok) handledAuthRequestsRef.current.add(request.toolCallId);
    }
  }, [pendingAuthorizations, agentAccessMode]);

  const respondToAuthorization = (toolCallId: string, permissionOptionId: string) =>
    NativeZedLLM.respondToToolAuthorization(toolCallId, permissionOptionId);

  const clearCompletedToolCalls = () => {};

  return {
    toolCalls: toolCalls as ToolCallInfo[],
    pendingAuthorizations: pendingAuthorizations as ToolAuthorizationRequest[],
    respondToAuthorization,
    clearCompletedToolCalls,
  };
}

