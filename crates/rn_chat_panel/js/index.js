import {AppRegistry} from 'react-native';
import {RNChatPanel} from './src/RNChatPanel';
import {RNChatHistoryPanel} from './src/RNChatHistoryPanel';

// Register the chat panel component with the module name expected by Zed
AppRegistry.registerComponent('RNChatPanel', () => RNChatPanel);
AppRegistry.registerComponent('RNChatHistoryPanel', () => RNChatHistoryPanel);

export default RNChatPanel;
