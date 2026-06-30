// Ambient declarations for packages that ship without TypeScript types.
// `skipLibCheck: true` in tsconfig skips type-checking inside node_modules,
// but we still need module declarations for TypeScript to accept the imports.
declare module "react-plotly.js";
declare module "plotly.js-dist-min";
