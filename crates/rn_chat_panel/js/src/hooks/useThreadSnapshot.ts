import { useProjection } from '@rngpui/app';
import NativeZedLLM, { type ThreadSnapshotPayload } from '../../../craby-modules/src/NativeZedLLM';

// Re-export types for consumers
export interface ThreadSnapshot {
  session_id: string;
  title: string;
  status: 'idle' | 'generating';
  entries: EntrySnapshot[];
  token_usage?: { input_tokens: number; output_tokens: number };
}

export type EntrySnapshot =
  | { type: 'user_message'; id?: string; content: string }
  | { type: 'assistant_message'; chunks: AssistantChunkSnapshot[] }
  | {
      type: 'tool_call';
      id: string;
      title: string;
      kind: string;
      status: string;
      input_json?: string;
      result_text?: string;
      error_message?: string;
      diff_count: number;
      terminal_count: number;
      authorization_options?: PermissionOptionSnapshot[];
    };

export type AssistantChunkSnapshot =
  | { type: 'text'; content: string }
  | { type: 'thinking'; content: string };

export interface PermissionOptionSnapshot {
  id: string;
  label: string;
  description?: string;
  is_default: boolean;
}

/**
 * Hook to consume AcpThread snapshots from native.
 * Uses the generic useProjection hook from @rngpui/app.
 */
export function useThreadSnapshot(): ThreadSnapshot | null {
  return useProjection<ThreadSnapshot, ThreadSnapshotPayload>(
    NativeZedLLM as any,
    'threadSnapshot',
    NativeZedLLM.threadSnapshot
  );
}
