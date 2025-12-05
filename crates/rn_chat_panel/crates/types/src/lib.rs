//! Shared types for rn_chat_panel and zed_modules.
//!
//! This crate exists to break the cyclic dependency between rn_chat_panel
//! (which handles GPUI integration) and zed_modules (which handles craby signals).

use std::sync::{Arc, Mutex, OnceLock};
use std::collections::{HashMap, HashSet};

use async_channel::{Receiver, Sender, bounded, unbounded};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};

// ============================================================================
// Data Types
// ============================================================================

/// Theme data exposed to React Native.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ThemeData {
    pub name: String,
    pub appearance: String, // "light" or "dark"
    pub colors: ThemeColorsData,
    pub fonts: FontSettingsData,
}

/// Subset of Zed's theme colors exposed to React Native.
/// Colors are hex strings (#RRGGBBAA).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ThemeColorsData {
    // Backgrounds
    pub background: String,
    pub surface_background: String,
    pub elevated_surface_background: String,
    pub panel_background: String,

    // Element states
    pub element_background: String,
    pub element_hover: String,
    pub element_active: String,
    pub element_selected: String,
    pub element_disabled: String,

    // Ghost element states
    pub ghost_element_background: String,
    pub ghost_element_hover: String,
    pub ghost_element_active: String,
    pub ghost_element_selected: String,

    // Text
    pub text: String,
    pub text_muted: String,
    pub text_placeholder: String,
    pub text_disabled: String,
    pub text_accent: String,

    // Icons
    pub icon: String,
    pub icon_muted: String,
    pub icon_disabled: String,
    pub icon_accent: String,

    // Borders
    pub border: String,
    pub border_variant: String,
    pub border_focused: String,
    pub border_selected: String,
    pub border_disabled: String,

    // UI elements
    pub tab_bar_background: String,
    pub tab_active_background: String,
    pub tab_inactive_background: String,
    pub status_bar_background: String,
    pub title_bar_background: String,
    pub toolbar_background: String,
    pub scrollbar_thumb_background: String,
}

/// Font settings exposed to React Native.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FontSettingsData {
    pub ui_font_family: String,
    pub ui_font_size: f32,
    pub buffer_font_family: String,
    pub buffer_font_size: f32,
}

/// Workspace info exposed to React Native.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct WorkspaceInfo {
    pub project_name: Option<String>,
    pub root_path: Option<String>,
    pub current_file_path: Option<String>,
}

// ============================================================================
// Language Model Types
// ============================================================================

/// Language model info exposed to React Native.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LanguageModelInfo {
    pub id: String,
    pub name: String,
    pub provider_id: String,
    pub provider_name: String,
    pub supports_images: bool,
    pub supports_tools: bool,
}

/// Language model provider info exposed to React Native.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LanguageModelProviderInfo {
    pub id: String,
    pub name: String,
    pub is_authenticated: bool,
}

// ============================================================================
// Agent Server Types
// ============================================================================

/// Agent server type enum matching the TypeScript side.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum AgentServerType {
    Codex,
    Claude,
    Gemini,
    Custom,
}

impl std::fmt::Display for AgentServerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentServerType::Codex => write!(f, "codex"),
            AgentServerType::Claude => write!(f, "claude"),
            AgentServerType::Gemini => write!(f, "gemini"),
            AgentServerType::Custom => write!(f, "custom"),
        }
    }
}

impl AgentServerType {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "codex" => Some(AgentServerType::Codex),
            "claude" => Some(AgentServerType::Claude),
            "gemini" => Some(AgentServerType::Gemini),
            "custom" => Some(AgentServerType::Custom),
            _ => None,
        }
    }
}

/// Agent server info exposed to React Native.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentServerInfo {
    #[serde(rename = "type")]
    pub agent_type: AgentServerType,
    pub name: String,
    pub is_connected: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_name: Option<String>,
}

/// Agent model info exposed to React Native (for agent server model selection UIs).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RnAgentModelInfo {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub selected: bool,
}

/// Agent mode info exposed to React Native (for permission/mode selection UIs).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RnAgentModeInfo {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub selected: bool,
}

// ============================================================================
// ACP Thread History Types
// ============================================================================

/// ACP thread metadata for listing (lightweight, for history sidebar).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AcpThreadMetadata {
    pub id: String,
    pub title: String,
    pub updated_at: String, // ISO 8601
}

/// Token usage statistics.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RnTokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

// ============================================================================
// Thread Snapshot Types
// ============================================================================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThreadSnapshot {
    pub session_id: String,
    pub title: String,
    pub status: ThreadStatusSnapshot,
    pub entries: Vec<EntrySnapshot>,
    pub token_usage: Option<RnTokenUsage>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThreadStatusSnapshot {
    Idle,
    Generating,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EntrySnapshot {
    UserMessage {
        id: Option<String>,
        content: String,
    },
    AssistantMessage {
        chunks: Vec<AssistantChunkSnapshot>,
    },
    ToolCall {
        id: String,
        title: String,
        kind: String,
        status: String,
        input_json: Option<String>,
        result_text: Option<String>,
        error_message: Option<String>,
        diff_count: u32,
        terminal_count: u32,
        authorization_options: Option<Vec<PermissionOptionSnapshot>>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AssistantChunkSnapshot {
    Text { content: String },
    Thinking { content: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PermissionOptionSnapshot {
    pub id: String,
    pub label: String,
    pub description: Option<String>,
    pub is_default: bool,
}

// ============================================================================
// Parsed Markdown Types (for React Native rendering)
// ============================================================================

/// Top-level parsed markdown structure.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RnParsedMarkdown {
    pub elements: Vec<RnMarkdownElement>,
}

/// A parsed markdown element.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum RnMarkdownElement {
    Heading {
        level: u8,
        content: Vec<RnMarkdownInline>,
    },
    Paragraph {
        content: Vec<RnMarkdownInline>,
    },
    CodeBlock {
        language: Option<String>,
        code: String,
        highlights: Vec<RnCodeHighlight>,
    },
    BlockQuote {
        children: Vec<RnMarkdownElement>,
    },
    List {
        ordered: bool,
        start: Option<u64>,
        items: Vec<RnListItem>,
    },
    HorizontalRule,
    Table {
        headers: Vec<Vec<RnMarkdownInline>>,
        rows: Vec<Vec<Vec<RnMarkdownInline>>>,
        alignments: Vec<String>,
    },
    Image {
        alt: String,
        url: String,
    },
}

/// A list item in markdown.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RnListItem {
    pub item_type: RnListItemType,
    pub content: Vec<RnMarkdownElement>,
}

/// Type of list item.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum RnListItemType {
    Ordered { number: u64 },
    Unordered,
    Task { checked: bool },
}

/// Inline markdown content (text with styling).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum RnMarkdownInline {
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        style: Option<RnTextStyle>,
    },
    Code {
        code: String,
    },
    Link {
        text: String,
        url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        style: Option<RnTextStyle>,
    },
    Image {
        alt: String,
        url: String,
    },
}

/// Text styling for inline markdown.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RnTextStyle {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bold: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub italic: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strikethrough: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub underline: Option<bool>,
}

impl RnTextStyle {
    pub fn is_empty(&self) -> bool {
        self.bold.is_none()
            && self.italic.is_none()
            && self.strikethrough.is_none()
            && self.underline.is_none()
    }
}

/// Syntax highlighting for code blocks.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RnCodeHighlight {
    pub start: usize,
    pub end: usize,
    /// Hex color string (#RRGGBB or #RRGGBBAA)
    pub color: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub font_weight: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub font_style: Option<String>,
}

// ============================================================================
// Tool Call Types
// ============================================================================

/// The kind of tool being called.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolKind {
    Read,
    Write,
    Execute,
    Network,
    Other,
}

/// Status of a tool call.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallStatus {
    Pending,
    WaitingForConfirmation,
    InProgress,
    Completed,
    Failed,
    Rejected,
    Canceled,
}

/// Information about a tool call.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolCallInfo {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<u64>,
    pub name: String,
    pub title: String,
    pub kind: ToolKind,
    pub status: ToolCallStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_json: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// Number of diffs available for this tool call (for write operations).
    #[serde(default)]
    pub diff_count: u32,
    /// Number of terminals available for this tool call (for execute operations).
    #[serde(default)]
    pub terminal_count: u32,
}

/// A permission option for tool authorization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PermissionOption {
    pub id: String,
    pub label: String,
    pub description: Option<String>,
    pub is_default: bool,
}

/// Tool authorization request.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolAuthorizationRequest {
    pub tool_call_id: String,
    pub title: String,
    pub options: Vec<PermissionOption>,
}

/// Tool call event types for signals.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolCallEvent {
    Started {
        tool_call: ToolCallInfo,
    },
    StatusChanged {
        id: String,
        status: ToolCallStatus,
    },
    ResultReady {
        id: String,
        result_text: String,
    },
    AuthorizationRequired {
        request: ToolAuthorizationRequest,
    },
    Error {
        id: String,
        error: String,
    },
    /// Emitted when native content (diffs/terminals) is updated for a tool call.
    ContentUpdated {
        id: String,
        diff_count: u32,
        terminal_count: u32,
    },
}

// ============================================================================
// Command Enum
// ============================================================================

/// Zed-specific commands that run on the GPUI app thread with App context.
pub enum ZedHostCommand {
    // Theme commands
    GetTheme {
        reply: Sender<ThemeData>,
    },
    GetColor {
        name: String,
        reply: Sender<Option<String>>,
    },

    // Workspace commands
    GetWorkspaceInfo {
        reply: Sender<WorkspaceInfo>,
    },

    // LLM commands - basic
    IsLlmAvailable {
        reply: Sender<bool>,
    },
    SendLlmMessage {
        module_id: usize,
        message: String,
        reply: Sender<u64>,
    },
    CancelLlmRequest {
        request_id: u64,
    },

    // LLM commands - extended for model selection
    GetProviders {
        reply: Sender<Vec<LanguageModelProviderInfo>>,
    },
    GetModels {
        reply: Sender<Vec<LanguageModelInfo>>,
    },
    GetDefaultModelId {
        reply: Sender<Option<String>>,
    },
    SendLlmMessageToModel {
        module_id: usize,
        model_id: String,
        message: String,
        reply: Sender<u64>,
    },

    // Agent server commands
    GetAgentServers {
        reply: Sender<Vec<AgentServerInfo>>,
    },
    ConnectAgent {
        agent_type: AgentServerType,
        reply: Sender<bool>,
    },
    DisconnectAgent {
        agent_type: AgentServerType,
    },
    SendAgentMessage {
        module_id: usize,
        agent_type: AgentServerType,
        message: String,
        reply: Sender<u64>,
    },

    // Agent configuration (per-agent, best-effort)
    EnsureAgentThread {
        agent_type: AgentServerType,
        reply: Sender<bool>,
    },
    StartAgentThread {
        agent_type: AgentServerType,
        reply: Sender<bool>,
    },
    GetAgentModels {
        agent_type: AgentServerType,
        reply: Sender<Vec<RnAgentModelInfo>>,
    },
    SetAgentModel {
        agent_type: AgentServerType,
        model_id: String,
        reply: Sender<bool>,
    },
    GetAgentModes {
        agent_type: AgentServerType,
        reply: Sender<Vec<RnAgentModeInfo>>,
    },
    SetAgentMode {
        agent_type: AgentServerType,
        mode_id: String,
        reply: Sender<bool>,
    },

    // Theme signal registration
    RegisterThemeListener {
        module_id: usize,
    },
    UnregisterThemeListener {
        module_id: usize,
    },

    // Workspace signal registration
    RegisterWorkspaceListener {
        module_id: usize,
    },
    UnregisterWorkspaceListener {
        module_id: usize,
    },

    // Agent connection signal registration
    RegisterAgentListener {
        module_id: usize,
    },
    UnregisterAgentListener {
        module_id: usize,
    },

    // ACP thread history commands
    ListAcpThreads {
        reply: Sender<Vec<AcpThreadMetadata>>,
    },
    SearchAcpThreads {
        query: String,
        reply: Sender<Vec<AcpThreadMetadata>>,
    },
    OpenAcpThread {
        thread_id: String,
        reply: Sender<bool>,
    },

    // ACP thread history signal registration
    RegisterAcpThreadsListener {
        module_id: usize,
    },
    UnregisterAcpThreadsListener {
        module_id: usize,
    },

    // Tool call commands
    RespondToToolAuthorization {
        tool_call_id: String,
        permission_option_id: String,
        reply: Sender<bool>,
    },
    CancelToolCall {
        tool_call_id: String,
        reply: Sender<bool>,
    },

    // Tool call signal registration
    RegisterToolCallListener {
        module_id: usize,
    },
    UnregisterToolCallListener {
        module_id: usize,
    },

    // Thread snapshot signal registration
    RegisterThreadSnapshotListener {
        module_id: usize,
    },
    UnregisterThreadSnapshotListener {
        module_id: usize,
    },

    // Markdown parsing
    ParseMarkdown {
        markdown: String,
        reply: Sender<RnParsedMarkdown>,
    },

    // UI commands (host-controlled layout)
    SetChatPaneVisible {
        visible: bool,
    },
    MinimizeChatPane,

    // Icon commands
    GetIconSvg {
        name: String,
        reply: Sender<String>,
    },
    ListIcons {
        reply: Sender<Vec<String>>,
    },

}

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug)]
pub enum CommandError {
    NotInitialized,
    ShuttingDown,
    ReceiverGone,
}

impl std::fmt::Display for CommandError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CommandError::NotInitialized => write!(f, "Command bus not initialized"),
            CommandError::ShuttingDown => write!(f, "Command bus is shutting down"),
            CommandError::ReceiverGone => write!(f, "Command receiver dropped"),
        }
    }
}

impl std::error::Error for CommandError {}

// ============================================================================
// Message Bus Infrastructure
// ============================================================================

enum Command {
    Zed(ZedHostCommand),
    Shutdown,
}

struct Inner {
    sender: async_channel::Sender<Command>,
    shutdown: AtomicBool,
}

impl Inner {
    fn is_shutting_down(&self) -> bool {
        self.shutdown.load(Ordering::SeqCst)
    }
}

#[derive(Clone)]
struct CommandSender {
    inner: Arc<Inner>,
}

impl CommandSender {
    fn send(&self, command: Command) -> Result<(), CommandError> {
        if self.inner.is_shutting_down() {
            return Err(CommandError::ShuttingDown);
        }
        self.inner
            .sender
            .send_blocking(command)
            .map_err(|_| CommandError::ReceiverGone)
    }
}

static BUS: OnceLock<Arc<Inner>> = OnceLock::new();

// ============================================================================
// Theme Listener Registry
// ============================================================================

/// Registry of active ZedTheme module instances that want theme change notifications.
static THEME_LISTENERS: OnceLock<Mutex<HashSet<usize>>> = OnceLock::new();

pub fn theme_listeners() -> &'static Mutex<HashSet<usize>> {
    THEME_LISTENERS.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Callback type for emitting theme changed signals.
static THEME_SIGNAL_EMITTER: OnceLock<Box<dyn Fn(usize, ThemeData) + Send + Sync>> = OnceLock::new();

/// Cache of the last emitted theme name for deduplication.
static LAST_THEME_NAME: OnceLock<Mutex<String>> = OnceLock::new();

fn last_theme_name() -> &'static Mutex<String> {
    LAST_THEME_NAME.get_or_init(|| Mutex::new(String::new()))
}

/// Set the callback for emitting theme changed signals.
pub fn set_theme_signal_emitter(emitter: Box<dyn Fn(usize, ThemeData) + Send + Sync>) {
    THEME_SIGNAL_EMITTER.set(emitter).ok();
}

/// Emit theme changed signal to all registered listeners.
/// Only emits if the theme name has changed.
pub fn emit_theme_changed(theme_data: ThemeData) {
    // Check if theme actually changed
    let should_emit = {
        let mut last = last_theme_name().lock().unwrap();
        if *last == theme_data.name {
            false
        } else {
            *last = theme_data.name.clone();
            true
        }
    };

    if !should_emit {
        return;
    }

    if let Some(emitter) = THEME_SIGNAL_EMITTER.get() {
        let listeners = theme_listeners().lock().unwrap();
        for &module_id in listeners.iter() {
            emitter(module_id, theme_data.clone());
        }
    }
}

// ============================================================================
// Workspace Listener Registry
// ============================================================================

/// Registry of active ZedWorkspace module instances that want workspace change notifications.
static WORKSPACE_LISTENERS: OnceLock<Mutex<HashSet<usize>>> = OnceLock::new();

pub fn workspace_listeners() -> &'static Mutex<HashSet<usize>> {
    WORKSPACE_LISTENERS.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Callback type for emitting workspace changed signals.
static WORKSPACE_SIGNAL_EMITTER: OnceLock<Box<dyn Fn(usize, WorkspaceInfo) + Send + Sync>> = OnceLock::new();

/// Set the callback for emitting workspace changed signals.
pub fn set_workspace_signal_emitter(emitter: Box<dyn Fn(usize, WorkspaceInfo) + Send + Sync>) {
    WORKSPACE_SIGNAL_EMITTER.set(emitter).ok();
}

/// Emit workspace changed signal to all registered listeners.
pub fn emit_workspace_changed(workspace_info: WorkspaceInfo) {
    if let Some(emitter) = WORKSPACE_SIGNAL_EMITTER.get() {
        let listeners = workspace_listeners().lock().unwrap();
        for &module_id in listeners.iter() {
            emitter(module_id, workspace_info.clone());
        }
    }
}

// ============================================================================
// ACP Thread History Listener Registry
// ============================================================================

/// Registry of active ZedLLM module instances that want ACP thread history updates.
static ACP_THREADS_LISTENERS: OnceLock<Mutex<HashSet<usize>>> = OnceLock::new();

pub fn acp_threads_listeners() -> &'static Mutex<HashSet<usize>> {
    ACP_THREADS_LISTENERS.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Callback type for emitting ACP threads changed signals.
static ACP_THREADS_SIGNAL_EMITTER: OnceLock<Box<dyn Fn(usize, String) + Send + Sync>> =
    OnceLock::new();

/// Set the callback for emitting ACP threads changed signals.
pub fn set_acp_threads_signal_emitter(emitter: Box<dyn Fn(usize, String) + Send + Sync>) {
    ACP_THREADS_SIGNAL_EMITTER.set(emitter).ok();
}

/// Emit ACP threads changed signal to all registered listeners.
pub fn emit_acp_threads_changed(threads: Vec<AcpThreadMetadata>) {
    let payload = serde_json::to_string(&threads).unwrap_or_else(|_| "[]".to_string());
    if let Some(emitter) = ACP_THREADS_SIGNAL_EMITTER.get() {
        let listeners = acp_threads_listeners().lock().unwrap();
        for &module_id in listeners.iter() {
            emitter(module_id, payload.clone());
        }
    }
}

// ============================================================================
// Tool Call Listener Registry
// ============================================================================

/// Registry of active module instances that want tool call notifications.
static TOOL_CALL_LISTENERS: OnceLock<Mutex<HashSet<usize>>> = OnceLock::new();

pub fn tool_call_listeners() -> &'static Mutex<HashSet<usize>> {
    TOOL_CALL_LISTENERS.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Callback type for emitting tool call event signals.
static TOOL_CALL_EVENT_EMITTER: OnceLock<Box<dyn Fn(usize, ToolCallEvent) + Send + Sync>> = OnceLock::new();

/// Set the callback for emitting tool call event signals.
pub fn set_tool_call_signal_emitter(emitter: Box<dyn Fn(usize, ToolCallEvent) + Send + Sync>) {
    TOOL_CALL_EVENT_EMITTER.set(emitter).ok();
}

/// Emit tool call event signal to all registered listeners.
pub fn emit_tool_call_event(event: ToolCallEvent) {
    if let Some(emitter) = TOOL_CALL_EVENT_EMITTER.get() {
        let listeners = tool_call_listeners().lock().unwrap();
        for &module_id in listeners.iter() {
            emitter(module_id, event.clone());
        }
    }
}

// ============================================================================
// Thread Snapshot Listener Registry
// ============================================================================

static THREAD_SNAPSHOT_EMITTER: OnceLock<Box<dyn Fn(usize, ThreadSnapshot) + Send + Sync>> =
    OnceLock::new();
static THREAD_SNAPSHOT_LISTENERS: OnceLock<Mutex<HashSet<usize>>> = OnceLock::new();

pub fn set_thread_snapshot_emitter(emitter: Box<dyn Fn(usize, ThreadSnapshot) + Send + Sync>) {
    let _ = THREAD_SNAPSHOT_EMITTER.set(emitter);
}

pub fn thread_snapshot_listeners() -> &'static Mutex<HashSet<usize>> {
    THREAD_SNAPSHOT_LISTENERS.get_or_init(|| Mutex::new(HashSet::new()))
}

pub fn emit_thread_snapshot(module_id: usize, snapshot: ThreadSnapshot) {
    if let Some(emitter) = THREAD_SNAPSHOT_EMITTER.get() {
        let listeners = thread_snapshot_listeners().lock().unwrap();
        if listeners.is_empty() {
            return;
        }
        if listeners.contains(&module_id) {
            emitter(module_id, snapshot.clone());
        }
        for &listener_id in listeners.iter() {
            if listener_id != module_id {
                emitter(listener_id, snapshot.clone());
            }
        }
    }
}

// ============================================================================
// LLM Request Registry
// ============================================================================

/// Tracks active LLM streaming requests for cancellation and signal routing.
pub struct LlmRequestState {
    pub module_id: usize,
    pub cancel_tx: Sender<()>,
}

static LLM_REQUESTS: OnceLock<Mutex<HashMap<u64, LlmRequestState>>> = OnceLock::new();

pub fn llm_requests() -> &'static Mutex<HashMap<u64, LlmRequestState>> {
    LLM_REQUESTS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Callback type for emitting LLM chunk signals.
static LLM_CHUNK_EMITTER: OnceLock<Box<dyn Fn(usize, u64, String) + Send + Sync>> = OnceLock::new();
/// Callback type for emitting LLM done signals.
static LLM_DONE_EMITTER: OnceLock<Box<dyn Fn(usize, u64) + Send + Sync>> = OnceLock::new();
/// Callback type for emitting LLM error signals.
static LLM_ERROR_EMITTER: OnceLock<Box<dyn Fn(usize, u64, String) + Send + Sync>> = OnceLock::new();

/// Set the callbacks for emitting LLM signals.
pub fn set_llm_signal_emitters(
    chunk: Box<dyn Fn(usize, u64, String) + Send + Sync>,
    done: Box<dyn Fn(usize, u64) + Send + Sync>,
    error: Box<dyn Fn(usize, u64, String) + Send + Sync>,
) {
    LLM_CHUNK_EMITTER.set(chunk).ok();
    LLM_DONE_EMITTER.set(done).ok();
    LLM_ERROR_EMITTER.set(error).ok();
}

/// Get the LLM chunk emitter if set.
pub fn llm_chunk_emitter() -> Option<&'static Box<dyn Fn(usize, u64, String) + Send + Sync>> {
    LLM_CHUNK_EMITTER.get()
}

/// Get the LLM done emitter if set.
pub fn llm_done_emitter() -> Option<&'static Box<dyn Fn(usize, u64) + Send + Sync>> {
    LLM_DONE_EMITTER.get()
}

/// Get the LLM error emitter if set.
pub fn llm_error_emitter() -> Option<&'static Box<dyn Fn(usize, u64, String) + Send + Sync>> {
    LLM_ERROR_EMITTER.get()
}

// ============================================================================
// Agent Server Signal Emitters
// ============================================================================

/// Callback type for emitting agent connected signals.
static AGENT_CONNECTED_EMITTER: OnceLock<Box<dyn Fn(usize, AgentServerType, String) + Send + Sync>> = OnceLock::new();
/// Callback type for emitting agent disconnected signals.
static AGENT_DISCONNECTED_EMITTER: OnceLock<Box<dyn Fn(usize, AgentServerType, String) + Send + Sync>> = OnceLock::new();

/// Set the callbacks for emitting agent signals.
pub fn set_agent_signal_emitters(
    connected: Box<dyn Fn(usize, AgentServerType, String) + Send + Sync>,
    disconnected: Box<dyn Fn(usize, AgentServerType, String) + Send + Sync>,
) {
    AGENT_CONNECTED_EMITTER.set(connected).ok();
    AGENT_DISCONNECTED_EMITTER.set(disconnected).ok();
}

/// Get the agent connected emitter if set.
pub fn agent_connected_emitter() -> Option<&'static Box<dyn Fn(usize, AgentServerType, String) + Send + Sync>> {
    AGENT_CONNECTED_EMITTER.get()
}

/// Get the agent disconnected emitter if set.
pub fn agent_disconnected_emitter() -> Option<&'static Box<dyn Fn(usize, AgentServerType, String) + Send + Sync>> {
    AGENT_DISCONNECTED_EMITTER.get()
}

// ============================================================================
// Agent Listener Registry
// ============================================================================

/// Registry of module instances that want agent connection notifications.
static AGENT_LISTENERS: OnceLock<Mutex<HashSet<usize>>> = OnceLock::new();

pub fn agent_listeners() -> &'static Mutex<HashSet<usize>> {
    AGENT_LISTENERS.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Emit agent connected signal to all registered listeners.
pub fn emit_agent_connected(agent_type: AgentServerType, session_id: String) {
    if let Some(emitter) = agent_connected_emitter() {
        let listeners = agent_listeners().lock().unwrap();
        for &module_id in listeners.iter() {
            emitter(module_id, agent_type.clone(), session_id.clone());
        }
    }
}

/// Emit agent disconnected signal to all registered listeners.
pub fn emit_agent_disconnected(agent_type: AgentServerType, session_id: String) {
    if let Some(emitter) = agent_disconnected_emitter() {
        let listeners = agent_listeners().lock().unwrap();
        for &module_id in listeners.iter() {
            emitter(module_id, agent_type.clone(), session_id.clone());
        }
    }
}

// ============================================================================
// Bus Initialization (called from rn_chat_panel)
// ============================================================================

/// Initialize the command bus. Returns a receiver for processing commands.
/// This is called by rn_chat_panel during initialization.
pub fn init_bus() -> Option<Receiver<ZedHostCommand>> {
    if BUS.get().is_some() {
        return None;
    }

    let (sender, receiver) = unbounded();
    let inner = Arc::new(Inner {
        sender,
        shutdown: AtomicBool::new(false),
    });

    if BUS.set(inner).is_ok() {
        // Create a channel adapter that unwraps Command::Zed
        let (zed_tx, zed_rx) = unbounded();

        // Spawn a task to forward commands (this will be driven by the caller)
        std::thread::spawn(move || {
            while let Ok(cmd) = receiver.recv_blocking() {
                match cmd {
                    Command::Zed(zed_cmd) => {
                        if zed_tx.send_blocking(zed_cmd).is_err() {
                            break;
                        }
                    }
                    Command::Shutdown => break,
                }
            }
        });

        Some(zed_rx)
    } else {
        None
    }
}

fn sender() -> Result<CommandSender, CommandError> {
    BUS.get()
        .map(|inner| CommandSender { inner: inner.clone() })
        .ok_or(CommandError::NotInitialized)
}

// ============================================================================
// Public API
// ============================================================================

/// Send a Zed host command (fire-and-forget).
pub fn send(cmd: ZedHostCommand) -> Result<(), CommandError> {
    sender()?.send(Command::Zed(cmd))
}

/// Send a Zed host command that expects a reply (async).
pub async fn send_with_reply<T>(
    build: impl FnOnce(Sender<T>) -> ZedHostCommand,
) -> Result<T, CommandError> {
    let (tx, rx) = bounded(1);
    send(build(tx))?;
    rx.recv().await.map_err(|_| CommandError::ReceiverGone)
}

/// Send a Zed host command that expects a reply (blocking).
/// Use this from TurboModule sync methods.
pub fn send_with_reply_blocking<T>(
    build: impl FnOnce(Sender<T>) -> ZedHostCommand,
) -> Result<T, CommandError> {
    let (tx, rx) = bounded(1);
    send(build(tx))?;
    rx.recv_blocking().map_err(|_| CommandError::ReceiverGone)
}

/// Shutdown the command bus.
pub fn shutdown() {
    if let Some(inner) = BUS.get() {
        if inner.shutdown.swap(true, Ordering::SeqCst) {
            return;
        }
        let _ = inner.sender.send_blocking(Command::Shutdown);
    }
}
