import React, { useEffect, useRef } from 'react';
import { Animated, View, type ViewStyle } from 'react-native';
import { useZedTheme } from '../theme';

export type SpinnerSize = 'small' | 'default' | 'large';

export interface SpinnerProps {
  size?: SpinnerSize;
  color?: 'default' | 'accent' | 'muted';
  style?: ViewStyle;
}

const SIZE_MAP: Record<SpinnerSize, number> = {
  small: 14,
  default: 20,
  large: 32,
};

export function Spinner({ size = 'default', color = 'default', style }: SpinnerProps) {
  const { colors } = useZedTheme();
  const rotation = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const animation = Animated.loop(
      Animated.timing(rotation, {
        toValue: 1,
        duration: 800,
        useNativeDriver: true,
      })
    );
    animation.start();
    return () => animation.stop();
  }, [rotation]);

  const spinnerSize = SIZE_MAP[size];
  const borderWidth = size === 'small' ? 1.5 : size === 'large' ? 3 : 2;

  const spinnerColor = {
    default: colors.icon,
    accent: colors.iconAccent,
    muted: colors.iconMuted,
  }[color];

  const rotateInterpolate = rotation.interpolate({
    inputRange: [0, 1],
    outputRange: ['0deg', '360deg'],
  });

  return (
    <View style={[{ width: spinnerSize, height: spinnerSize }, style]}>
      <Animated.View
        style={{
          width: spinnerSize,
          height: spinnerSize,
          borderRadius: spinnerSize / 2,
          borderWidth,
          borderColor: colors.border,
          borderTopColor: spinnerColor,
          transform: [{ rotate: rotateInterpolate }],
        }}
      />
    </View>
  );
}
