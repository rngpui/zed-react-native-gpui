import React from 'react';
import { View, Image, type ViewStyle, type ImageSourcePropType } from 'react-native';
import { Text } from './Text';
import { useZedTheme } from '../theme';

export type AvatarSize = 'small' | 'default' | 'medium' | 'large';

export interface AvatarProps {
  src?: ImageSourcePropType | string;
  name?: string;
  size?: AvatarSize;
  style?: ViewStyle;
}

const SIZE_MAP: Record<AvatarSize, { container: number; fontSize: number }> = {
  small: { container: 20, fontSize: 10 },
  default: { container: 28, fontSize: 12 },
  medium: { container: 36, fontSize: 14 },
  large: { container: 48, fontSize: 18 },
};

function getInitials(name: string): string {
  const parts = name.trim().split(/\s+/);
  if (parts.length === 1) {
    return parts[0].charAt(0).toUpperCase();
  }
  return (parts[0].charAt(0) + parts[parts.length - 1].charAt(0)).toUpperCase();
}

function stringToColor(str: string): string {
  // Generate a deterministic color from a string
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }
  const hue = Math.abs(hash) % 360;
  return `hsl(${hue}, 50%, 40%)`;
}

export function Avatar({ src, name, size = 'default', style }: AvatarProps) {
  const { colors } = useZedTheme();
  const { container: containerSize, fontSize } = SIZE_MAP[size];

  const hasImage = !!src;
  const initials = name ? getInitials(name) : '?';
  const bgColor = name ? stringToColor(name) : colors.elementBackground;

  const imageSource = typeof src === 'string' ? { uri: src } : src;

  return (
    <View
      style={[
        {
          width: containerSize,
          height: containerSize,
          borderRadius: containerSize / 2,
          backgroundColor: hasImage ? colors.surfaceBackground : bgColor,
          alignItems: 'center',
          justifyContent: 'center',
          overflow: 'hidden',
        },
        style,
      ]}
    >
      {hasImage ? (
        <Image
          source={imageSource!}
          style={{
            width: containerSize,
            height: containerSize,
            borderRadius: containerSize / 2,
          }}
          resizeMode="cover"
        />
      ) : (
        <Text
          style={{
            fontSize,
            fontWeight: '600',
            color: '#fff',
          }}
        >
          {initials}
        </Text>
      )}
    </View>
  );
}
