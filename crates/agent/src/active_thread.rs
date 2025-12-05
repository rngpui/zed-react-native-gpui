use acp_thread::AcpThread;
use collections::HashMap;
use gpui::{App, AppContext, Context, Entity, EntityId, EventEmitter, Global, WeakEntity};
use project::Project;

pub enum ActiveAcpThreadEvent {
    Changed { project_id: EntityId },
}

pub struct ActiveAcpThreadStore {
    active_by_project: HashMap<EntityId, WeakEntity<AcpThread>>,
}

impl EventEmitter<ActiveAcpThreadEvent> for ActiveAcpThreadStore {}

struct GlobalActiveAcpThreadStore(Entity<ActiveAcpThreadStore>);

impl Global for GlobalActiveAcpThreadStore {}

impl ActiveAcpThreadStore {
    pub fn global(cx: &mut App) -> Entity<Self> {
        if cx.has_global::<GlobalActiveAcpThreadStore>() {
            return cx.global::<GlobalActiveAcpThreadStore>().0.clone();
        }

        let store = cx.new(|_| ActiveAcpThreadStore {
            active_by_project: HashMap::default(),
        });
        cx.set_global(GlobalActiveAcpThreadStore(store.clone()));
        store
    }

    pub fn active_thread(&self, project_id: EntityId) -> Option<Entity<AcpThread>> {
        self.active_by_project.get(&project_id)?.upgrade()
    }

    pub fn set_active_thread(
        &mut self,
        project: &Entity<Project>,
        thread: Option<Entity<AcpThread>>,
        cx: &mut Context<Self>,
    ) {
        let project_id = project.entity_id();
        let new_weak = thread.map(|thread| thread.downgrade());

        let should_update = match (self.active_by_project.get(&project_id), new_weak.as_ref()) {
            (None, None) => false,
            (Some(existing), Some(next)) => existing.entity_id() != next.entity_id(),
            _ => true,
        };

        if !should_update {
            return;
        }

        match new_weak {
            Some(weak) => {
                self.active_by_project.insert(project_id, weak);
            }
            None => {
                self.active_by_project.remove(&project_id);
            }
        }

        cx.emit(ActiveAcpThreadEvent::Changed { project_id });
        cx.notify();
    }
}
