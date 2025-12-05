import React, { createContext, useContext, useState, useEffect, type ReactNode } from 'react';
import { View, ActivityIndicator } from 'react-native';
import type { ZedTheme } from './types';
import { transformNativeTheme } from './transform';

const ZedThemeContext = createContext<ZedTheme | null>(null);

interface ZedThemeProviderProps {
  children: ReactNode;
}

/**
 * Provides Zed theme context to child components.
 * Waits for native theme to be available before rendering children.
 */
export function ZedThemeProvider({ children }: ZedThemeProviderProps) {
  const [theme, setTheme] = useState<ZedTheme | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    try {
      const nativeTheme = getNativeTheme();
      if (nativeTheme) {
        setTheme(nativeTheme);
      } else {
        setError('Native ZedTheme module not available');
      }

      const subscription = subscribeToThemeChanges((data) => {
        setTheme(transformNativeTheme(data));
      });
      return () => subscription?.unsubscribe();
    } catch (e) {
      console.error('Failed to initialize theme:', e);
    }
  }, []);

  // Show loading state while waiting for native theme
  if (!theme) {
    return (
      <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: '#1a1a1a' }}>
        {error ? (
          <View style={{ padding: 20 }}>
            <ActivityIndicator color="#ff6b6b" />
          </View>
        ) : (
          <ActivityIndicator color="#6b8afd" />
        )}
      </View>
    );
  }

  return <ZedThemeContext.Provider value={theme}>{children}</ZedThemeContext.Provider>;
}

export function useZedTheme(): ZedTheme {
  const theme = useContext(ZedThemeContext);
  if (!theme) {
    throw new Error('useZedTheme must be used within a ZedThemeProvider with loaded theme');
  }
  return theme;
}

export function useZedColors() {
  return useZedTheme().colors;
}

export function useZedFonts() {
  return useZedTheme().fonts;
}

// Native module integration
type NativeThemeData = Parameters<typeof transformNativeTheme>[0];
let _nativeZedTheme: any = null;

function getNativeZedTheme() {
  if (_nativeZedTheme !== null) return _nativeZedTheme;
  try {
    const { NativeModuleRegistry } = require('craby-modules');
    _nativeZedTheme = NativeModuleRegistry.get('ZedTheme');
  } catch (e) {
    console.error('Failed to get ZedTheme module:', e);
    _nativeZedTheme = undefined;
  }
  return _nativeZedTheme;
}

function getNativeTheme(): ZedTheme | null {
  const mod = getNativeZedTheme();
  if (mod?.getTheme) {
    try {
      return transformNativeTheme(mod.getTheme());
    } catch (e) {
      console.error('Failed to get theme:', e);
    }
  }
  return null;
}

function subscribeToThemeChanges(callback: (data: NativeThemeData) => void) {
  const mod = getNativeZedTheme();
  if (mod?.themeChanged) {
    try {
      // Register listener with Rust first - this triggers RegisterThemeListener command
      mod.addListener('themeChanged');
      // Then subscribe to the signal (stores callback in C++)
      const unsubscribe = mod.themeChanged((e: { theme: NativeThemeData }) => {
        callback(e.theme);
      });
      return {
        unsubscribe: () => {
          unsubscribe();
          mod.removeListeners(1);
        }
      };
    } catch (e) {
      console.error('Failed to subscribe to theme changes:', e);
    }
  }
  return null;
}
