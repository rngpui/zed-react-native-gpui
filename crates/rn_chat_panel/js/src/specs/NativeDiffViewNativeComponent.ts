/**
 * NativeDiffView - Renders a diff editor for a tool call.
 *
 * This component displays the file diff associated with a tool call.
 * The actual diff content is looked up on the native side using
 * the toolCallId and diffIndex props.
 *
 * @format
 */

import type {ViewProps} from 'react-native';
import type {Int32} from 'react-native/Libraries/Types/CodegenTypes';
import codegenNativeComponent from 'react-native/Libraries/Utilities/codegenNativeComponent';
import type {HostComponent} from 'react-native';

export interface NativeDiffViewProps extends ViewProps {
  /**
   * The tool call ID to look up the diff content.
   * Required - the component won't render without this.
   */
  toolCallId: string;

  /**
   * Index of the diff within the tool call's diffs array.
   * Defaults to 0 (first diff).
   */
  diffIndex?: Int32;
}

export default codegenNativeComponent<NativeDiffViewProps>(
  'NativeDiffView',
) as HostComponent<NativeDiffViewProps>;
