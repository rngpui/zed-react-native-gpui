import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useLayoutEffect,
  useRef,
  type ReactNode,
} from 'react';
import {
  View,
  Pressable,
  ScrollView,
  StyleSheet,
  Animated,
} from 'react-native';
import OverlayHostView from '@rngpui/app/Libraries/Modal/OverlayHostViewNativeComponent.gpui';
import { Text } from '../components/Text';
import { useZedTheme } from '../theme';
import { radii, spacing } from '../tokens';

export interface DropdownItem {
  id: string;
  title: string;
  subtitle?: string;
}

interface DropdownMenuState {
  dropdownId: string;
  items: DropdownItem[];
  onSelect: (id: string) => void;
  layout: {
    placement: 'up' | 'down';
    left: number;
    top?: number;
    bottom?: number;
    minWidth: number;
    maxHeight: number;
  };
}

type DropdownContextValue = {
  openDropdownId: string | null;
  setOpenDropdownId: (id: string | null) => void;
  showDropdownMenu: (state: DropdownMenuState) => void;
  hideDropdownMenu: () => void;
};

const DropdownContext = createContext<DropdownContextValue | null>(null);

const ANIMATION_DURATION = 150;

interface DropdownMenuItemProps {
  item: DropdownItem;
  onSelect: (id: string) => void;
  isLast: boolean;
}

function DropdownMenuItem({ item, onSelect, isLast }: DropdownMenuItemProps) {
  const { colors } = useZedTheme();
  const [hovered, setHovered] = useState(false);
  const hoverAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(hoverAnim, {
      toValue: hovered ? 1 : 0,
      duration: 100,
      useNativeDriver: false,
    }).start();
  }, [hovered, hoverAnim]);

  const backgroundColor = hoverAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ['transparent', colors.elementHover],
  });

  return (
    <Pressable
      onPress={() => onSelect(item.id)}
      onHoverIn={() => setHovered(true)}
      onHoverOut={() => setHovered(false)}
    >
      <Animated.View
        style={[
          styles.option,
          {
            backgroundColor,
            borderBottomWidth: isLast ? 0 : StyleSheet.hairlineWidth,
            borderBottomColor: colors.border,
          },
        ]}
      >
        <Text size="sm" color="default">
          {item.title}
        </Text>
        {item.subtitle && (
          <Text size="xs" color="muted" style={{ marginTop: 2 }}>
            {item.subtitle}
          </Text>
        )}
      </Animated.View>
    </Pressable>
  );
}

export function DropdownProvider({ children }: { children: ReactNode }) {
  const [openDropdownId, setOpenDropdownId] = useState<string | null>(null);
  const [menuState, setMenuState] = useState<DropdownMenuState | null>(null);
  const [isVisible, setIsVisible] = useState(false);
  const [shouldAnimateIn, setShouldAnimateIn] = useState(false);
  const { colors } = useZedTheme();

  const backdropOpacity = useRef(new Animated.Value(0)).current;
  const dropdownOpacity = useRef(new Animated.Value(0)).current;
  const dropdownScale = useRef(new Animated.Value(0.95)).current;

  // Start animation after overlay elements are mounted
  useLayoutEffect(() => {
    if (shouldAnimateIn && isVisible && menuState) {
      const handle = requestAnimationFrame(() => {
        Animated.parallel([
          Animated.timing(backdropOpacity, {
            toValue: 1,
            duration: ANIMATION_DURATION,
            useNativeDriver: true,
          }),
          Animated.timing(dropdownOpacity, {
            toValue: 1,
            duration: ANIMATION_DURATION,
            useNativeDriver: true,
          }),
          Animated.timing(dropdownScale, {
            toValue: 1,
            duration: ANIMATION_DURATION,
            useNativeDriver: true,
          }),
        ]).start(() => {
          setShouldAnimateIn(false);
        });
      });

      return () => {
        cancelAnimationFrame(handle);
      };
    }
  }, [shouldAnimateIn, isVisible, menuState, backdropOpacity, dropdownOpacity, dropdownScale]);

  const animateOut = (callback?: () => void) => {
    Animated.parallel([
      Animated.timing(backdropOpacity, {
        toValue: 0,
        duration: ANIMATION_DURATION,
        useNativeDriver: true,
      }),
      Animated.timing(dropdownOpacity, {
        toValue: 0,
        duration: ANIMATION_DURATION,
        useNativeDriver: true,
      }),
      Animated.timing(dropdownScale, {
        toValue: 0.95,
        duration: ANIMATION_DURATION,
        useNativeDriver: true,
      }),
    ]).start(() => {
      setIsVisible(false);
      callback?.();
    });
  };

  const showDropdownMenu = (state: DropdownMenuState) => {
    backdropOpacity.setValue(0);
    dropdownOpacity.setValue(0);
    dropdownScale.setValue(0.95);
    setMenuState(state);
    setOpenDropdownId(state.dropdownId);
    setIsVisible(true);
    setShouldAnimateIn(true);
  };

  const hideDropdownMenu = () => {
    animateOut(() => {
      setMenuState(null);
      setOpenDropdownId(null);
    });
  };

  const handleSelect = (id: string) => {
    if (menuState) {
      menuState.onSelect(id);
    }
    hideDropdownMenu();
  };

  const handleBackdropPress = () => {
    hideDropdownMenu();
  };

  const dropdownStyle = menuState
    ? {
        position: 'absolute' as const,
        left: menuState.layout.left,
        ...(menuState.layout.top !== undefined
          ? { top: menuState.layout.top }
          : { bottom: menuState.layout.bottom }),
        minWidth: menuState.layout.minWidth,
        maxHeight: menuState.layout.maxHeight,
        backgroundColor: colors.elementBackground,
        borderRadius: radii.md,
        borderWidth: 1,
        borderColor: colors.border,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.15,
        shadowRadius: 8,
        overflow: 'hidden' as const,
      }
    : {};

  return (
    <DropdownContext.Provider
      value={{ openDropdownId, setOpenDropdownId, showDropdownMenu, hideDropdownMenu }}
    >
      <View style={styles.container}>
        {children}

        {/* Portal layer for dropdown menus - rendered at window level */}
        {isVisible && menuState && (
          <OverlayHostView
            visible={true}
            animationType="none"
            presentationStyle="overFullScreen"
            transparent={true}
            pointerEvents="box-none"
            style={StyleSheet.absoluteFillObject}
          >
            <View pointerEvents="box-none" style={StyleSheet.absoluteFillObject}>
              {/* Invisible backdrop to capture taps outside dropdown */}
              <Animated.View
                style={[
                  styles.backdrop,
                  {
                    opacity: backdropOpacity,
                    backgroundColor: 'transparent',
                  },
                ]}
              >
                <Pressable
                  style={StyleSheet.absoluteFillObject}
                  onPress={handleBackdropPress}
                />
              </Animated.View>

              {/* Dropdown menu */}
              <Animated.View
                style={[
                  dropdownStyle,
                  {
                    opacity: dropdownOpacity,
                    transform: [{ scale: dropdownScale }],
                  },
                ]}
              >
                {menuState.items.length === 0 ? (
                  <View style={styles.emptyState}>
                    <Text size="sm" color="muted">
                      No options available
                    </Text>
                  </View>
                ) : (
                  <ScrollView
                    style={{ maxHeight: menuState.layout.maxHeight }}
                    showsVerticalScrollIndicator={false}
                    bounces={false}
                  >
                    {menuState.items.map((item, index) => (
                      <DropdownMenuItem
                        key={item.id}
                        item={item}
                        onSelect={handleSelect}
                        isLast={index === menuState.items.length - 1}
                      />
                    ))}
                  </ScrollView>
                )}
              </Animated.View>
            </View>
          </OverlayHostView>
        )}
      </View>
    </DropdownContext.Provider>
  );
}

export function useDropdownContext(): DropdownContextValue {
  const context = useContext(DropdownContext);
  if (!context) {
    throw new Error('useDropdownContext must be used within a DropdownProvider');
  }
  return context;
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  backdrop: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
  },
  option: {
    paddingVertical: spacing.px2,
    paddingHorizontal: spacing.px3,
  },
  emptyState: {
    padding: spacing.px3,
  },
});
