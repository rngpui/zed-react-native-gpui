import React, { useState, useEffect, useRef } from 'react';
import { Pressable, View, Animated, type PressableProps, type ViewStyle } from 'react-native';
import { Text } from './Text';
import { useZedTheme } from '../theme';

export interface ToggleProps extends Omit<PressableProps, 'style' | 'children'> {
  enabled: boolean;
  onEnabledChange?: (enabled: boolean) => void;
  label?: string;
  disabled?: boolean;
  size?: 'default' | 'small';
  style?: ViewStyle;
}

export function Toggle({
  enabled,
  onEnabledChange,
  label,
  disabled = false,
  size = 'default',
  style,
  ...props
}: ToggleProps) {
  const { colors } = useZedTheme();
  const [hovered, setHovered] = useState(false);
  const translateX = useRef(new Animated.Value(enabled ? 1 : 0)).current;

  const trackWidth = size === 'small' ? 32 : 40;
  const trackHeight = size === 'small' ? 18 : 22;
  const thumbSize = size === 'small' ? 14 : 18;
  const thumbMargin = 2;

  useEffect(() => {
    Animated.timing(translateX, {
      toValue: enabled ? 1 : 0,
      duration: 150,
      useNativeDriver: true,
    }).start();
  }, [enabled, translateX]);

  const getTrackColor = () => {
    if (disabled) return colors.elementDisabled;
    if (enabled) return colors.textAccent;
    if (hovered) return colors.elementHover;
    return colors.elementBackground;
  };

  const thumbTranslateX = translateX.interpolate({
    inputRange: [0, 1],
    outputRange: [thumbMargin, trackWidth - thumbSize - thumbMargin],
  });

  return (
    <Pressable
      disabled={disabled}
      onPress={() => onEnabledChange?.(!enabled)}
      onHoverIn={() => setHovered(true)}
      onHoverOut={() => setHovered(false)}
      style={[{ flexDirection: 'row', alignItems: 'center', gap: 8 }, style]}
      {...props}
    >
      <View
        style={{
          width: trackWidth,
          height: trackHeight,
          borderRadius: trackHeight / 2,
          backgroundColor: getTrackColor(),
          justifyContent: 'center',
        }}
      >
        <Animated.View
          style={{
            width: thumbSize,
            height: thumbSize,
            borderRadius: thumbSize / 2,
            backgroundColor: disabled ? colors.textDisabled : colors.background,
            transform: [{ translateX: thumbTranslateX }],
            shadowColor: '#000',
            shadowOffset: { width: 0, height: 1 },
            shadowOpacity: 0.15,
            shadowRadius: 2,
          }}
        />
      </View>
      {label && (
        <Text color={disabled ? 'disabled' : 'default'}>{label}</Text>
      )}
    </Pressable>
  );
}
