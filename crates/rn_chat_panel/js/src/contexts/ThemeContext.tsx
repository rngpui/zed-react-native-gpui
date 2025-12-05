import React, { createContext, useContext, type ReactNode } from 'react';
import { useZedTheme } from '@zed/ui';

type ZedTheme = ReturnType<typeof useZedTheme>;

const ThemeContext = createContext<ZedTheme | null>(null);

export function ThemeProvider({ children }: { children: ReactNode }) {
  const theme = useZedTheme();
  return (
    <ThemeContext.Provider value={theme}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme(): ZedTheme {
  const theme = useContext(ThemeContext);
  if (!theme) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return theme;
}

export function useColors() {
  return useTheme().colors;
}

export function useFonts() {
  return useTheme().fonts;
}
