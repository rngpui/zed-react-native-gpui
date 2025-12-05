import type { NativeModule } from 'craby-modules';
import { NativeModuleRegistry } from 'craby-modules';

// ============================================================================
// TurboModule Spec
// ============================================================================

export interface Spec extends NativeModule {
  // Get the SVG content for a given icon name
  // Returns empty string if icon not found
  getIconSvg(name: string): string;

  // Get all available icon names as JSON array
  listIcons(): string;
}

export default NativeModuleRegistry.getEnforcing<Spec>('ZedIcons');
