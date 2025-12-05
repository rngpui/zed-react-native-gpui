use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::{Mutex, OnceLock};

use acp_thread::{AcpThread, AgentConnection, AgentThreadEntry};
use agent::{ActiveAcpThreadEvent, ActiveAcpThreadStore, HistoryEntry, HistoryStore};
use gpui::{App, AppContext, Context, Entity, EntityId, EventEmitter, Subscription, WeakEntity};
use project::Project;
use rn_chat_panel_types::{AcpThreadMetadata, AgentServerType, emit_acp_threads_changed};

// RefCell is !Sync, so use Mutex (still only touched on GPUI thread).
static RUNTIMES: OnceLock<Mutex<HashMap<EntityId, Entity<RnChatRuntime>>>> = OnceLock::new();

fn runtimes() -> &'static Mutex<HashMap<EntityId, Entity<RnChatRuntime>>> {
    RUNTIMES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Key for agent connections. Currently just wraps AgentServerType,
/// but extensible if we need to support multiple instances per type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AgentKey {
    pub agent_type: AgentServerType,
    // Future: pub name: Option<String>, for custom/extension agents.
}

impl From<AgentServerType> for AgentKey {
    fn from(agent_type: AgentServerType) -> Self {
        Self { agent_type }
    }
}

pub struct RnChatRuntime {
    /// Live agent connections by key.
    connections: HashMap<AgentKey, Rc<dyn AgentConnection>>,

    /// Active thread per agent key (strong refs to keep thread alive).
    active_threads: HashMap<AgentKey, Entity<AcpThread>>,

    /// Tool call resolution: tool_call_id -> thread that owns it.
    tool_call_threads: HashMap<String, WeakEntity<AcpThread>>,

    /// Known tool call IDs per thread (for incremental registration).
    known_tool_calls: HashMap<EntityId, HashSet<String>>,

    /// Subscriptions to active threads (for tool call registration).
    _thread_subscriptions: HashMap<EntityId, Subscription>,

    /// History store for ACP thread listings.
    history_store: Option<WeakEntity<HistoryStore>>,
    _history_subscription: Option<Subscription>,

    /// Sync with native active thread selection.
    _active_thread_subscription: Option<Subscription>,

    /// Project reference.
    project: WeakEntity<Project>,
}

pub enum RnChatRuntimeEvent {
    ConnectionChanged { agent_key: AgentKey, connected: bool },
    ActiveThreadChanged { agent_key: AgentKey },
    ToolCallsUpdated { thread_id: EntityId },
}

impl EventEmitter<RnChatRuntimeEvent> for RnChatRuntime {}

impl RnChatRuntime {
    /// Get or create runtime for a project. Keyed by EntityId.
    /// NOTE: If Project is recreated, a new runtime is created (acceptable).
    pub fn get_or_create(project: &Entity<Project>, cx: &mut App) -> Entity<RnChatRuntime> {
        let project_id = project.entity_id();
        let mut runtimes = runtimes().lock().unwrap();

        runtimes.retain(|_, runtime| runtime.read(cx).project().is_some());

        if let Some(runtime) = runtimes.get(&project_id) {
            return runtime.clone();
        }

        let runtime = cx.new(|_| RnChatRuntime {
            connections: HashMap::new(),
            active_threads: HashMap::new(),
            tool_call_threads: HashMap::new(),
            known_tool_calls: HashMap::new(),
            _thread_subscriptions: HashMap::new(),
            history_store: None,
            _history_subscription: None,
            _active_thread_subscription: None,
            project: project.downgrade(),
        });

        runtimes.insert(project_id, runtime.clone());
        runtime
    }

    // === Connection Management ===

    pub fn connection(&self, key: &AgentKey) -> Option<Rc<dyn AgentConnection>> {
        self.connections.get(key).cloned()
    }

    pub fn set_connection(
        &mut self,
        key: AgentKey,
        conn: Rc<dyn AgentConnection>,
        cx: &mut Context<Self>,
    ) {
        self.connections.insert(key.clone(), conn);
        cx.emit(RnChatRuntimeEvent::ConnectionChanged { agent_key: key, connected: true });
        cx.notify();
    }

    pub fn remove_connection(&mut self, key: &AgentKey, cx: &mut Context<Self>) {
        if self.connections.remove(key).is_some() {
            cx.emit(RnChatRuntimeEvent::ConnectionChanged {
                agent_key: key.clone(),
                connected: false,
            });
            cx.notify();
        }
    }

    pub fn is_connected(&self, key: &AgentKey) -> bool {
        self.connections.contains_key(key)
    }

    // === Thread Management ===

    pub fn active_thread(&self, key: &AgentKey) -> Option<Entity<AcpThread>> {
        self.active_threads.get(key).cloned()
    }

    pub fn set_active_thread(
        &mut self,
        key: AgentKey,
        thread: Entity<AcpThread>,
        cx: &mut Context<Self>,
    ) {
        if let Some(old_thread) = self.active_threads.remove(&key) {
            let old_id = old_thread.entity_id();
            self._thread_subscriptions.remove(&old_id);
            self.prune_tool_calls_for_thread(old_id);
        }

        let thread_id = thread.entity_id();
        let subscription = cx.observe(&thread, |runtime, thread, cx| {
            runtime.update_tool_calls_for_thread(&thread, cx);
        });
        self._thread_subscriptions.insert(thread_id, subscription);

        self.update_tool_calls_for_thread(&thread, cx);

        self.active_threads.insert(key.clone(), thread);
        cx.emit(RnChatRuntimeEvent::ActiveThreadChanged { agent_key: key });
        cx.notify();
    }

    pub fn ensure_active_thread_sync(&mut self, cx: &mut Context<Self>) {
        if self._active_thread_subscription.is_some() {
            return;
        }

        let Some(project) = self.project.upgrade() else {
            return;
        };

        let project_id = project.entity_id();
        let store = ActiveAcpThreadStore::global(cx);

        let subscription = cx.subscribe(&store, move |runtime, store, event, cx| {
            let ActiveAcpThreadEvent::Changed { project_id: changed_id } = event;
            if *changed_id != project_id {
                return;
            }
            runtime.sync_active_thread_from_store(&store, project_id, cx);
        });

        self._active_thread_subscription = Some(subscription);
        self.sync_active_thread_from_store(&store, project_id, cx);
    }

    fn sync_active_thread_from_store(
        &mut self,
        store: &Entity<ActiveAcpThreadStore>,
        project_id: EntityId,
        cx: &mut Context<Self>,
    ) {
        let Some(thread) = store.read(cx).active_thread(project_id) else {
            return;
        };
        self.maybe_set_active_thread(thread, cx);
    }

    fn maybe_set_active_thread(&mut self, thread: Entity<AcpThread>, cx: &mut Context<Self>) {
        let key = AgentKey::from(AgentServerType::Custom);
        if let Some(current) = self.active_threads.get(&key) {
            if current.entity_id() == thread.entity_id() {
                return;
            }
        }
        self.set_active_thread(key, thread, cx);
    }

    pub fn clear_active_thread(&mut self, key: &AgentKey, cx: &mut Context<Self>) {
        if let Some(thread) = self.active_threads.remove(key) {
            let thread_id = thread.entity_id();
            self._thread_subscriptions.remove(&thread_id);
            self.prune_tool_calls_for_thread(thread_id);
            cx.emit(RnChatRuntimeEvent::ActiveThreadChanged { agent_key: key.clone() });
            cx.notify();
        }
    }

    // === Tool Call Resolution (Incremental) ===

    /// Update tool call index for a thread. Called from thread subscription.
    fn update_tool_calls_for_thread(&mut self, thread: &Entity<AcpThread>, cx: &mut Context<Self>) {
        let thread_id = thread.entity_id();
        let known = self.known_tool_calls.entry(thread_id).or_default();
        let weak = thread.downgrade();

        let mut updated = false;
        for entry in thread.read(cx).entries() {
            if let AgentThreadEntry::ToolCall(call) = entry {
                let id = call.id.to_string();
                if !known.contains(&id) {
                    self.tool_call_threads.insert(id.clone(), weak.clone());
                    known.insert(id);
                    updated = true;
                }
            }
        }

        if updated {
            cx.emit(RnChatRuntimeEvent::ToolCallsUpdated { thread_id });
        }
    }

    pub fn thread_for_tool_call(&self, tool_call_id: &str) -> Option<Entity<AcpThread>> {
        self.tool_call_threads.get(tool_call_id)?.upgrade()
    }

    fn prune_tool_calls_for_thread(&mut self, thread_id: EntityId) {
        if let Some(known) = self.known_tool_calls.remove(&thread_id) {
            for id in known {
                self.tool_call_threads.remove(&id);
            }
        }
    }

    // === ACP Thread History ===

    pub fn set_history_store(&mut self, store: Entity<HistoryStore>, cx: &mut Context<Self>) {
        let should_replace = match &self.history_store {
            Some(existing) => existing
                .upgrade()
                .map_or(true, |existing| existing.entity_id() != store.entity_id()),
            None => true,
        };

        if !should_replace {
            return;
        }

        let subscription = cx.observe(&store, |runtime, store, cx| {
            runtime.emit_acp_threads_from_store(&store, cx);
        });
        self._history_subscription = Some(subscription);
        self.history_store = Some(store.downgrade());
        self.emit_acp_threads_from_store(&store, cx);
    }

    pub fn acp_threads(&self, cx: &App) -> Option<Vec<AcpThreadMetadata>> {
        let store = self.history_store.as_ref()?.upgrade()?;
        Some(Self::collect_acp_threads(store.read(cx)))
    }

    fn emit_acp_threads_from_store(
        &self,
        store: &Entity<HistoryStore>,
        cx: &mut Context<Self>,
    ) {
        let threads = store.read_with(cx, |store, _| Self::collect_acp_threads(store));
        emit_acp_threads_changed(threads);
    }

    fn collect_acp_threads(store: &HistoryStore) -> Vec<AcpThreadMetadata> {
        store
            .entries()
            .filter_map(|entry| match entry {
                HistoryEntry::AcpThread(thread) => Some(AcpThreadMetadata {
                    id: thread.id.to_string(),
                    title: thread.title.to_string(),
                    updated_at: thread.updated_at.to_rfc3339(),
                }),
                HistoryEntry::TextThread(_) => None,
            })
            .collect()
    }

    // === Project Access ===

    pub fn project(&self) -> Option<Entity<Project>> {
        self.project.upgrade()
    }
}
