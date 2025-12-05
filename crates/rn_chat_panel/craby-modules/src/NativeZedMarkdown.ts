import type { NativeModule } from 'craby-modules';
import { NativeModuleRegistry } from 'craby-modules';

// ============================================================================
// Module Spec
// ============================================================================

export interface Spec extends NativeModule {
  /**
   * Parse markdown text and return the parsed structure as JSON.
   */
  parse(markdown: string): string;
}

export default NativeModuleRegistry.getEnforcing<Spec>('ZedMarkdown');
