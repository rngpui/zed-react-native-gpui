const { makeGpuiBabelConfig } = require('@rngpui/app/tools/babel-config');

module.exports = makeGpuiBabelConfig({
  reactCompiler: true,
  reanimated: true,
});
