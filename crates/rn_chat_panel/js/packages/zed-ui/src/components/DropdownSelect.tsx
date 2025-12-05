import React, { useRef } from 'react';
import { View, Pressable, StyleSheet } from 'react-native';
import { Text } from './Text';
import { Icon, type IconColor } from './Icon';
import { useZedTheme } from '../theme';
import { useDropdownContext, type DropdownItem } from '../contexts/DropdownContext';
import { radii, sizing, spacing } from '../tokens';

type AnchorRect = { x: number; y: number; width: number; height: number };
type AnchoredLayout = {
  placement: 'up' | 'down';
  left: number;
  top?: number;
  bottom?: number;
  minWidth: number;
  maxHeight: number;
};

const { computeAnchoredOverlayLayout } = require('@rngpui/app/Libraries/Modal/AnchoredOverlay.gpui') as {
  computeAnchoredOverlayLayout: (args: {
    anchorRect: AnchorRect;
    placement?: 'auto' | 'up' | 'down';
    offset?: number;
    maxHeight?: number;
    minWidth?: number;
    clampToWindow?: boolean;
    windowBounds?: { width: number; height: number };
  }) => AnchoredLayout;
};

export type { DropdownItem as DropdownSelectItem } from '../contexts/DropdownContext';

interface DropdownSelectProps {
  dropdownId: string;
  icon?: string;
  placeholder: string;
  selectedId: string | null;
  items: DropdownItem[];
  onSelect: (id: string) => void;
  compact?: boolean;
  placement?: 'auto' | 'up' | 'down';
}

export function DropdownSelect({
  dropdownId,
  icon,
  placeholder,
  selectedId,
  items,
  onSelect,
  compact = true,
  placement = 'down',
}: DropdownSelectProps) {
  const { colors, fonts } = useZedTheme();
  const containerRef = useRef<any>(null);
  const { openDropdownId, showDropdownMenu, hideDropdownMenu } = useDropdownContext();
  const isOpen = openDropdownId === dropdownId;

  const selected = selectedId ? items.find((item) => item.id === selectedId) : undefined;
  const label = selected?.title ?? placeholder;

  const iconColorName: IconColor = isOpen ? 'accent' : 'muted';
  const textColor = isOpen ? colors.textAccent : colors.textMuted;
  const triggerBg = isOpen ? `${colors.textAccent}20` : 'transparent';

  const toggleOpen = () => {
    if (isOpen) {
      hideDropdownMenu();
      return;
    }

    const dropdownDesiredHeight = 300;
    const margin = spacing.px3;

    const measure = () => {
      if (!containerRef.current?.measureInWindow) {
        // Fallback if measurement isn't available
        const layout = computeAnchoredOverlayLayout({
          anchorRect: { x: 0, y: 0, width: 150, height: 32 },
          placement,
          offset: margin,
          maxHeight: dropdownDesiredHeight,
          minWidth: 150,
        });
        showDropdownMenu({ dropdownId, items, onSelect, layout });
        return;
      }

      containerRef.current.measureInWindow(
        (x: number, y: number, width: number, height: number) => {
          const layout = computeAnchoredOverlayLayout({
            anchorRect: { x, y, width, height },
            placement,
            offset: margin,
            maxHeight: dropdownDesiredHeight,
            minWidth: 150,
          });
          showDropdownMenu({ dropdownId, items, onSelect, layout });
        }
      );
    };

    requestAnimationFrame(measure);
  };

  return (
    <View ref={containerRef} style={styles.container}>
      <Pressable
        onPress={toggleOpen}
        style={[
          styles.trigger,
          compact && styles.triggerCompact,
          { backgroundColor: triggerBg },
        ]}
      >
        {icon ? <Icon name={icon} size={sizing.iconSm} color={iconColorName} /> : null}
        <Text
          numberOfLines={1}
          style={[
            styles.triggerText,
            { color: textColor, fontSize: fonts.ui.sm },
          ]}
        >
          {label}
        </Text>
        <Icon
          name={isOpen ? 'chevron-up' : 'chevron-down'}
          size={sizing.iconXs}
          color={iconColorName}
        />
      </Pressable>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    position: 'relative',
  },
  trigger: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.px1,
    paddingHorizontal: spacing.px1p5,
    paddingVertical: spacing.px1,
    borderRadius: radii.md,
  },
  triggerCompact: {},
  triggerText: {
    maxWidth: 140,
  },
});
