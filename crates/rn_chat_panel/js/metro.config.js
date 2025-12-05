/* eslint-disable import/no-commonjs */
'use strict';

const path = require('path');

function resolveFrom(moduleId, paths) {
  return require.resolve(moduleId, {paths});
}

function requireFrom(moduleId, paths) {
  return require(resolveFrom(moduleId, paths));
}

function escapeForRegex(str) {
  return String(str).replace(/[-/\\^$*+?.()|[\]{}]/g, r => `\\${r}`);
}

const appRoot = __dirname;

// Point to the actual RNGPUI packages location
const rngpuiRoot = path.resolve(appRoot, '..', '..', '..', '..', 'react-native-gpui');
const gpuiPkg = path.join(rngpuiRoot, 'packages', 'app');
const reactNativePkgRoot = path.join(appRoot, 'node_modules', 'react-native');

const {getDefaultConfig, mergeConfig} = requireFrom('@react-native/metro-config', [appRoot]);
const {withGPUIResolver} = requireFrom('@rngpui/metro-resolver', [
  appRoot,
  rngpuiRoot,
  gpuiPkg,
]);

const workspacePackages = {
  '@rngpui/app': gpuiPkg,
  '@rngpui/window': path.join(rngpuiRoot, 'packages', 'window'),
  '@rngpui/blur': path.join(rngpuiRoot, 'packages', 'blur'),
  '@rngpui/gradient': path.join(rngpuiRoot, 'packages', 'gradient'),
  '@rngpui/svg': path.join(rngpuiRoot, 'packages', 'svg'),
  '@rngpui/example-ui': path.join(rngpuiRoot, 'packages', 'example-ui'),
  '@rngpui/metro-resolver': path.join(rngpuiRoot, 'packages', 'metro-resolver'),
};

const extraNodeModules = {
  ...workspacePackages,
  'react-native': reactNativePkgRoot,
  react: path.join(appRoot, 'node_modules', 'react'),
  '@zed/ui': path.join(appRoot, 'packages', 'zed-ui'),
};

const baseConfig = getDefaultConfig(appRoot);
const {resolver: baseResolver} = baseConfig;
const defaultSourceExts = baseResolver?.sourceExts || [];

const config = mergeConfig(baseConfig, {
  projectRoot: appRoot,
  watchFolders: [
    path.join(rngpuiRoot, 'packages'),
    path.join(rngpuiRoot, 'node_modules'),
    path.join(appRoot, 'packages'), // Watch local @zed/ui package
    path.join(appRoot, '..', 'craby-modules'), // Watch Craby-generated TurboModule TS specs
  ],
  resolver: {
    extraNodeModules,
    nodeModulesPaths: [
      path.resolve(appRoot, 'node_modules'),
      path.join(rngpuiRoot, 'node_modules'),
    ],
    platforms: ['gpui', 'ios', 'native', 'android'],
    sourceExts: Array.from(new Set([...defaultSourceExts, 'ts', 'tsx', 'gpui.ts', 'gpui.tsx'])),
    blockList: new RegExp(
      [
        // Block react-native from RNGPUI to use Zed panel's version
        path.join(rngpuiRoot, 'packages', 'app', 'node_modules', 'react-native'),
        path.join(rngpuiRoot, 'node_modules', 'react-native'),
        // Block react from RNGPUI to prevent duplicate React instances
        path.join(rngpuiRoot, 'packages', 'app', 'node_modules', 'react'),
        path.join(rngpuiRoot, 'node_modules', 'react'),
        // Also block the pnpm store versions
        path.join(rngpuiRoot, 'node_modules', '.pnpm', 'react@'),
        path.join(rngpuiRoot, 'node_modules', '.pnpm', 'react-native@'),
      ]
        .map(p => `^${escapeForRegex(p)}.*$`)
        .join('|'),
    ),
  },
  serializer: {
    getModulesRunBeforeMainModule: () => [
      resolveFrom('react-native/Libraries/Core/InitializeCore', [appRoot]),
    ],
  },
});

const shims = {
  'expo-blur': {
    path: path.join(gpuiPkg, 'src', 'shims', 'expo-blur.gpui.tsx'),
  },
  'expo-linear-gradient': {
    path: resolveFrom('@rngpui/gradient', [appRoot, rngpuiRoot, gpuiPkg]),
  },
  'react-native-safe-area-context': {
    path: path.join(gpuiPkg, 'src', 'shims', 'react-native-safe-area-context.gpui.tsx'),
  },
  'react-native-screens': {
    path: path.join(gpuiPkg, 'src', 'shims', 'react-native-screens.gpui.tsx'),
  },
  'react-native-svg': {
    path: resolveFrom('@rngpui/svg', [appRoot, rngpuiRoot, gpuiPkg]),
  },
};

const gpuiConfig = withGPUIResolver(config, {
  reactNativeRoot: reactNativePkgRoot,
  gpuiRoot: gpuiPkg,
  shims,
});

const {wrapWithReanimatedMetroConfig} = requireFrom(
  'react-native-reanimated/metro-config',
  [appRoot],
);

module.exports = wrapWithReanimatedMetroConfig(gpuiConfig);
