import React from 'react';
import { View, Pressable, StyleSheet, Linking } from 'react-native';
import { Text } from './Text';
import { useZedTheme } from '../theme';
import { radii, spacing } from '../tokens';
import {
  parseMarkdown,
  type ParsedMarkdown,
  type MarkdownElement,
  type MarkdownInline,
  type TextStyle,
  type ListItem,
} from '../../../../../craby-modules/MarkdownTypes';

export interface MarkdownRendererProps {
  content: string;
}

type ThemeType = ReturnType<typeof useZedTheme>;

export function MarkdownRenderer({ content }: MarkdownRendererProps) {
  const parsed = React.useMemo(() => {
    try {
      return parseMarkdown(content);
    } catch {
      return { elements: [] } as ParsedMarkdown;
    }
  }, [content]);

  return (
    <View style={styles.container}>
      {parsed.elements.map((element, index) => (
        <MarkdownBlock key={index} element={element} />
      ))}
    </View>
  );
}

interface MarkdownBlockProps {
  element: MarkdownElement;
}

function MarkdownBlock({ element }: MarkdownBlockProps) {
  const { colors, fonts } = useZedTheme();

  const baseFontSize = fonts.uiFontSize ?? 14;

  switch (element.type) {
    case 'heading':
      return (
        <View style={styles.heading}>
          <Text
            style={[
              styles.headingText,
              getHeadingStyle(element.level, baseFontSize),
              { color: colors.text },
            ]}
          >
            <InlineContent content={element.content} />
          </Text>
        </View>
      );

    case 'paragraph':
      return (
        <View style={styles.paragraph}>
          <Text style={[styles.paragraphText, { color: colors.text }]}>
            <InlineContent content={element.content} />
          </Text>
        </View>
      );

    case 'codeBlock':
      return (
        <View
          style={[
            styles.codeBlock,
            { backgroundColor: colors.surfaceBackground, borderColor: colors.border },
          ]}
        >
          {element.language && (
            <Text style={[styles.codeLanguage, { color: colors.textMuted, fontSize: fonts.ui.xs }]}>
              {element.language}
            </Text>
          )}
          <Text
            style={[
              styles.codeText,
              {
                fontFamily: fonts.bufferFontFamily,
                fontSize: fonts.bufferFontSize,
                color: colors.text,
              },
            ]}
          >
            {element.code}
          </Text>
        </View>
      );

    case 'blockQuote':
      return (
        <View
          style={[
            styles.blockQuote,
            { borderLeftColor: colors.border, backgroundColor: colors.surfaceBackground },
          ]}
        >
          {element.children.map((child, index) => (
            <MarkdownBlock key={index} element={child} />
          ))}
        </View>
      );

    case 'list':
      return (
        <View style={styles.list}>
          {element.items.map((item, index) => (
            <ListItemComponent key={index} item={item} ordered={element.ordered} />
          ))}
        </View>
      );

    case 'horizontalRule':
      return <View style={[styles.horizontalRule, { backgroundColor: colors.border }]} />;

    case 'table':
      return (
        <View style={[styles.table, { borderColor: colors.border }]}>
          <View
            style={[
              styles.tableRow,
              { backgroundColor: `${colors.elementBackground}08` },
            ]}
          >
            {element.headers.map((header, index) => (
              <View key={index} style={[styles.tableCell, { borderColor: colors.border }]}>
                <Text style={[styles.tableHeaderText, { color: colors.text }]}>
                  <InlineContent content={header} />
                </Text>
              </View>
            ))}
          </View>
          {element.rows.map((row, rowIndex) => (
            <View key={rowIndex} style={styles.tableRow}>
              {row.map((cell, cellIndex) => (
                <View key={cellIndex} style={[styles.tableCell, { borderColor: colors.border }]}>
                  <Text style={{ color: colors.text }}>
                    <InlineContent content={cell} />
                  </Text>
                </View>
              ))}
            </View>
          ))}
        </View>
      );

    case 'image':
      return (
        <View style={styles.imageContainer}>
          <Text style={{ color: colors.textMuted }}>[Image: {element.alt || element.url}]</Text>
        </View>
      );

    default:
      return null;
  }
}

interface ListItemComponentProps {
  item: ListItem;
  ordered: boolean;
}

function ListItemComponent({ item }: ListItemComponentProps) {
  const { colors, fonts } = useZedTheme();

  const getBullet = () => {
    switch (item.itemType.type) {
      case 'ordered':
        return `${item.itemType.number}. `;
      case 'task':
        return item.itemType.checked ? '☑ ' : '☐ ';
      case 'unordered':
      default:
        return '• ';
    }
  };

  return (
    <View style={styles.listItem}>
      <Text style={[styles.listBullet, { color: colors.textMuted, fontSize: fonts.ui.md }]}>
        {getBullet()}
      </Text>
      <View style={styles.listContent}>
        {item.content.map((element, index) => (
          <MarkdownBlock key={index} element={element} />
        ))}
      </View>
    </View>
  );
}

interface InlineContentProps {
  content: MarkdownInline[];
}

function InlineContent({ content }: InlineContentProps) {
  const { colors } = useZedTheme();

  return (
    <>
      {content.map((inline, index) => (
        <InlineElement key={index} inline={inline} colors={colors} />
      ))}
    </>
  );
}

interface InlineElementProps {
  inline: MarkdownInline;
  colors: ThemeType['colors'];
}

function InlineElement({ inline, colors }: InlineElementProps) {
  const { fonts } = useZedTheme();

  switch (inline.type) {
    case 'text':
      return <Text style={getTextStyle(inline.style, colors)}>{inline.text}</Text>;

    case 'code':
      return (
        <Text
          style={[
            styles.inlineCode,
            {
              fontFamily: fonts.bufferFontFamily,
              fontSize: fonts.bufferFontSize,
              backgroundColor: colors.surfaceBackground,
              color: colors.textAccent,
            },
          ]}
        >
          {inline.code}
        </Text>
      );

    case 'link':
      return (
        <Pressable onPress={() => Linking.openURL(inline.url)}>
          <Text
            style={[
              { color: colors.textAccent, textDecorationLine: 'underline' },
              getTextStyle(inline.style, colors),
            ]}
          >
            {inline.text}
          </Text>
        </Pressable>
      );

    case 'image':
      return <Text style={{ color: colors.textMuted }}>[Image: {inline.alt || inline.url}]</Text>;

    default:
      return null;
  }
}

function getHeadingStyle(level: number, baseFontSize: number) {
  const scales = [1.75, 1.5, 1.25, 1.1, 1.0, 0.875];
  const scale = scales[Math.min(level - 1, scales.length - 1)] ?? 1.0;
  const fontSize = Math.round(baseFontSize * scale);

  return {
    fontSize,
    fontWeight: level <= 2 ? ('700' as const) : ('600' as const),
  };
}

function getTextStyle(style: TextStyle | undefined, colors: ThemeType['colors']) {
  if (!style) return { color: colors.text };

  return {
    color: colors.text,
    fontWeight: style.bold ? ('700' as const) : ('400' as const),
    fontStyle: style.italic ? ('italic' as const) : ('normal' as const),
    textDecorationLine: style.strikethrough
      ? ('line-through' as const)
      : style.underline
        ? ('underline' as const)
        : ('none' as const),
  };
}

const styles = StyleSheet.create({
  container: {
    gap: spacing.px2,
  },
  heading: {
    marginTop: spacing.px1,
  },
  headingText: {
    lineHeight: 28,
  },
  paragraph: {},
  paragraphText: {
    lineHeight: 22,
  },
  codeBlock: {
    borderRadius: radii.md,
    borderWidth: 1,
    padding: spacing.px3,
    overflow: 'hidden',
  },
  codeLanguage: {
    marginBottom: spacing.px2,
    textTransform: 'uppercase',
  },
  codeText: {
    lineHeight: 18,
  },
  blockQuote: {
    borderLeftWidth: 3,
    paddingLeft: spacing.px3,
    paddingVertical: spacing.px1,
    marginVertical: spacing.px1,
  },
  list: {
    gap: spacing.px1,
  },
  listItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  listBullet: {
    width: spacing.px5,
  },
  listContent: {
    flex: 1,
  },
  horizontalRule: {
    height: 1,
    marginVertical: spacing.px2,
  },
  table: {
    borderWidth: 1,
    borderRadius: radii.sm,
    overflow: 'hidden',
  },
  tableRow: {
    flexDirection: 'row',
  },
  tableCell: {
    flex: 1,
    padding: spacing.px2,
    borderWidth: 0.5,
  },
  tableHeaderText: {
    fontWeight: '600',
  },
  imageContainer: {
    padding: spacing.px2,
  },
  inlineCode: {
    borderRadius: radii.xs,
    paddingHorizontal: spacing.px1,
    paddingVertical: 1,
  },
});
