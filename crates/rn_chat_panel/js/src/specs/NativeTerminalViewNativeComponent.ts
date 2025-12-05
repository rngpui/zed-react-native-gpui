/**
 * NativeTerminalView - Renders a terminal view for a tool call.
 *
 * This component displays the terminal output associated with a tool call
 * (typically from an execute command). The actual terminal entity is looked
 * up on the native side using the toolCallId and terminalIndex props.
 *
 * @format
 */

import type {ViewProps} from 'react-native';
import type {Int32} from 'react-native/Libraries/Types/CodegenTypes';
import codegenNativeComponent from 'react-native/Libraries/Utilities/codegenNativeComponent';
import type {HostComponent} from 'react-native';

export interface NativeTerminalViewProps extends ViewProps {
  /**
   * The tool call ID to look up the terminal content.
   * Required - the component won't render without this.
   */
  toolCallId: string;

  /**
   * Index of the terminal within the tool call's terminals array.
   * Defaults to 0 (first terminal).
   */
  terminalIndex?: Int32;
}

export default codegenNativeComponent<NativeTerminalViewProps>(
  'NativeTerminalView',
) as HostComponent<NativeTerminalViewProps>;
