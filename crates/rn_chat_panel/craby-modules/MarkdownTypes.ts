// ============================================================================
// Parsed Markdown Types
// These types describe the JSON structure returned by NativeZedMarkdown.parse()
// ============================================================================

export type MarkdownElementType =
  | 'heading'
  | 'paragraph'
  | 'codeBlock'
  | 'blockQuote'
  | 'list'
  | 'horizontalRule'
  | 'table'
  | 'image';

export type MarkdownInlineType = 'text' | 'code' | 'link' | 'image';

export type ListItemType =
  | { type: 'ordered'; number: number }
  | { type: 'unordered' }
  | { type: 'task'; checked: boolean };

export interface TextStyle {
  bold?: boolean;
  italic?: boolean;
  strikethrough?: boolean;
  underline?: boolean;
}

export interface CodeHighlight {
  start: number;
  end: number;
  color: string;
  fontWeight?: string;
  fontStyle?: string;
}

export type MarkdownInline =
  | { type: 'text'; text: string; style?: TextStyle }
  | { type: 'code'; code: string }
  | { type: 'link'; text: string; url: string; style?: TextStyle }
  | { type: 'image'; alt: string; url: string };

export interface ListItem {
  itemType: ListItemType;
  content: MarkdownElement[];
}

export type MarkdownElement =
  | { type: 'heading'; level: number; content: MarkdownInline[] }
  | { type: 'paragraph'; content: MarkdownInline[] }
  | { type: 'codeBlock'; language?: string; code: string; highlights: CodeHighlight[] }
  | { type: 'blockQuote'; children: MarkdownElement[] }
  | { type: 'list'; ordered: boolean; start?: number; items: ListItem[] }
  | { type: 'horizontalRule' }
  | { type: 'table'; headers: MarkdownInline[][]; rows: MarkdownInline[][][]; alignments: string[] }
  | { type: 'image'; alt: string; url: string };

export interface ParsedMarkdown {
  elements: MarkdownElement[];
}

// ============================================================================
// Helper Functions
// ============================================================================

import NativeZedMarkdown from './src/NativeZedMarkdown';

/**
 * Parse markdown and return typed result.
 */
export function parseMarkdown(markdown: string): ParsedMarkdown {
  const json = NativeZedMarkdown.parse(markdown);
  return JSON.parse(json) as ParsedMarkdown;
}
