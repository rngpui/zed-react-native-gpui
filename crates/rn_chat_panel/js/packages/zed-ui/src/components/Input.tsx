import React, { useState } from 'react';
import { TextInput, type TextInputProps, type TextStyle } from 'react-native';
import { useZedTheme } from '../theme';

export interface InputProps extends Omit<TextInputProps, 'style'> {
  error?: boolean;
  style?: TextStyle;
}

export function Input({ error = false, style, onFocus, onBlur, editable = true, ...props }: InputProps) {
  const { colors, fonts } = useZedTheme();
  const [focused, setFocused] = useState(false);

  const borderColor = !editable ? colors.borderDisabled : error ? '#f85149' : focused ? colors.borderFocused : colors.border;
  const bgColor = !editable ? colors.elementDisabled : colors.surfaceBackground;

  return (
    <TextInput
      editable={editable}
      placeholderTextColor={colors.textPlaceholder}
      onFocus={(e) => { setFocused(true); onFocus?.(e); }}
      onBlur={(e) => { setFocused(false); onBlur?.(e); }}
      style={[{
        minHeight: 38,
        borderWidth: 1,
        borderColor,
        borderRadius: 6,
        paddingHorizontal: 10,
        backgroundColor: bgColor,
        color: editable ? colors.text : colors.textDisabled,
        fontSize: fonts.ui.sm,
        fontFamily: fonts.uiFontFamily,
      }, style]}
      {...props}
    />
  );
}
