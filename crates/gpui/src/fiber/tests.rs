    use super::{DirtyFlags, FiberSceneSegments, FiberTree, GlobalElementId};
    use crate::SceneSegmentId;
    use collections::FxHashSet;

    fn build_chain(tree: &mut FiberTree) -> (GlobalElementId, GlobalElementId, GlobalElementId) {
        let root = tree.create_placeholder_fiber();
        let child = tree.create_placeholder_fiber();
        let grandchild = tree.create_placeholder_fiber();
        tree.relink_children_in_order(&root, &[child]);
        tree.relink_children_in_order(&child, &[grandchild]);
        (root, child, grandchild)
    }

	    fn clear_dirty(tree: &mut FiberTree, ids: &[GlobalElementId]) {
	        for id in ids {
	            tree.set_dirty_flags(id, DirtyFlags::NONE);
	        }
	    }

    #[test]
    fn mark_dirty_layout_propagates_to_ancestors() {
        let mut tree = FiberTree::new();
        let (root, child, grandchild) = build_chain(&mut tree);
        clear_dirty(&mut tree, &[root, child, grandchild]);

        tree.mark_dirty(&grandchild, DirtyFlags::STRUCTURE_CHANGED);

	        let grandchild_dirty = tree.dirty_flags(&grandchild);
	        assert!(grandchild_dirty.contains(DirtyFlags::STRUCTURE_CHANGED));

	        let child_dirty = tree.dirty_flags(&child);
	        assert!(child_dirty.contains(DirtyFlags::HAS_LAYOUT_DIRTY_DESCENDANT));
	        assert!(!child_dirty.contains(DirtyFlags::HAS_DIRTY_SIZING_DESCENDANT));

	        let root_dirty = tree.dirty_flags(&root);
	        assert!(root_dirty.contains(DirtyFlags::HAS_LAYOUT_DIRTY_DESCENDANT));
	        assert!(!root_dirty.contains(DirtyFlags::HAS_DIRTY_SIZING_DESCENDANT));
    }

    #[test]
    fn mark_position_style_changed_propagates_layout_descendant() {
        let mut tree = FiberTree::new();
        let (root, child, grandchild) = build_chain(&mut tree);
        clear_dirty(&mut tree, &[root, child, grandchild]);

        tree.mark_position_style_changed(&grandchild);

	        let grandchild_dirty = tree.dirty_flags(&grandchild);
	        assert!(grandchild_dirty.contains(DirtyFlags::POSITION_STYLE_CHANGED));

	        let child_dirty = tree.dirty_flags(&child);
	        assert!(child_dirty.contains(DirtyFlags::HAS_LAYOUT_DIRTY_DESCENDANT));
	        assert!(!child_dirty.contains(DirtyFlags::HAS_PAINT_DIRTY_DESCENDANT));

	        let root_dirty = tree.dirty_flags(&root);
	        assert!(root_dirty.contains(DirtyFlags::HAS_LAYOUT_DIRTY_DESCENDANT));
	        assert!(!root_dirty.contains(DirtyFlags::HAS_PAINT_DIRTY_DESCENDANT));
    }

    #[test]
    fn remove_collects_scene_segments_for_subtree() {
        let mut tree = FiberTree::new();
        let root = tree.create_placeholder_fiber();
        let child = tree.create_placeholder_fiber();
        tree.relink_children_in_order(&root, &[child]);

        // Use SceneSegmentPool directly since Scene requires pool to be set
        let mut pool = crate::scene::SceneSegmentPool::default();
        let root_before = pool.alloc_segment();
        let root_after = pool.alloc_segment();
        let child_before = pool.alloc_segment();
        let child_after = pool.alloc_segment();

        tree.insert_scene_segments(
            &root,
            FiberSceneSegments {
                before: root_before,
                after: Some(root_after),
            },
        );
        tree.insert_scene_segments(
            &child,
            FiberSceneSegments {
                before: child_before,
                after: Some(child_after),
            },
        );

        tree.remove(&root);
        let removed = tree.take_removed_scene_segments();

        let removed_set: FxHashSet<SceneSegmentId> = removed.into_iter().collect();
        let expected_set: FxHashSet<SceneSegmentId> =
            vec![root_before, root_after, child_before, child_after]
                .into_iter()
                .collect();

        assert_eq!(removed_set, expected_set);
        assert!(tree.take_removed_scene_segments().is_empty());
    }

    #[test]
    fn create_and_get_placeholder_fiber() {
        let mut tree = FiberTree::new();
        let id = tree.create_placeholder_fiber();

	        let fiber = tree.get(&id);
	        assert!(fiber.is_some(), "Should be able to get created fiber");

	        assert!(
	            tree.dirty_flags(&id).contains(DirtyFlags::NEEDS_LAYOUT),
	            "New fiber should have NEEDS_LAYOUT flag"
	        );
	    }

    #[test]
    fn relink_children_updates_parent_child_relationships() {
        let mut tree = FiberTree::new();
        let parent_id = tree.create_placeholder_fiber();
        let child1 = tree.create_placeholder_fiber();
        let child2 = tree.create_placeholder_fiber();

        tree.relink_children_in_order(&parent_id, &[child1, child2]);

        // Check parent has both children
        let children = tree.children_slice(&parent_id);
        assert_eq!(children.len(), 2);
        assert_eq!(children[0], child1);
        assert_eq!(children[1], child2);

        // Check children have correct parent via tree method
        assert_eq!(tree.parent(&child1), Some(parent_id));
        assert_eq!(tree.parent(&child2), Some(parent_id));
    }

    #[test]
    fn mark_dirty_paint_propagates_correctly() {
        let mut tree = FiberTree::new();
        let (root, child, grandchild) = build_chain(&mut tree);
        clear_dirty(&mut tree, &[root, child, grandchild]);

        tree.mark_dirty(&grandchild, DirtyFlags::NEEDS_PAINT);

        // Grandchild should have NEEDS_PAINT
	        let grandchild_dirty = tree.dirty_flags(&grandchild);
	        assert!(grandchild_dirty.contains(DirtyFlags::NEEDS_PAINT));

	        // Parent should have HAS_DIRTY_DESCENDANT but not NEEDS_PAINT
	        let child_dirty = tree.dirty_flags(&child);
	        assert!(child_dirty.contains(DirtyFlags::HAS_PAINT_DIRTY_DESCENDANT));

	        // Root should also have HAS_DIRTY_DESCENDANT
	        let root_dirty = tree.dirty_flags(&root);
	        assert!(root_dirty.contains(DirtyFlags::HAS_PAINT_DIRTY_DESCENDANT));
    }

    #[test]
    fn remove_fiber_clears_from_tree() {
        let mut tree = FiberTree::new();
        let id = tree.create_placeholder_fiber();

        assert!(tree.get(&id).is_some());
        tree.remove(&id);
        assert!(tree.get(&id).is_none());
    }

    #[test]
    fn remove_fiber_removes_children_too() {
        let mut tree = FiberTree::new();
        let (root, child, grandchild) = build_chain(&mut tree);

        tree.remove(&root);

        assert!(tree.get(&root).is_none());
        assert!(tree.get(&child).is_none());
        assert!(tree.get(&grandchild).is_none());
    }

    #[test]
    fn dirty_flags_combine_correctly() {
        let flags = DirtyFlags::NEEDS_LAYOUT | DirtyFlags::NEEDS_PAINT;
        assert!(flags.contains(DirtyFlags::NEEDS_LAYOUT));
        assert!(flags.contains(DirtyFlags::NEEDS_PAINT));
        assert!(!flags.contains(DirtyFlags::CONTENT_CHANGED));
    }

    #[test]
    fn scene_segments_tracks_before_and_after() {
        let mut tree = FiberTree::new();
        let id = tree.create_placeholder_fiber();

        // Use SceneSegmentPool directly since Scene requires pool to be set
        let mut pool = crate::scene::SceneSegmentPool::default();
        let before = pool.alloc_segment();
        let after = pool.alloc_segment();

        tree.insert_scene_segments(
            &id,
            FiberSceneSegments {
                before,
                after: Some(after),
            },
        );

        let segments = tree.scene_segments(&id);
        assert!(segments.is_some());
        let segments = segments.unwrap();
        assert_eq!(segments.before, before);
        assert_eq!(segments.after, Some(after));
    }

    #[test]
    fn clear_dirty_flags_resets_flags() {
        let mut tree = FiberTree::new();
        let id = tree.create_placeholder_fiber();

        // Set some dirty flags
	        tree.mark_dirty(&id, DirtyFlags::NEEDS_LAYOUT | DirtyFlags::NEEDS_PAINT);

	        // Clear specific flags
	        let mut dirty = tree.dirty_flags(&id);
	        dirty.remove(DirtyFlags::NEEDS_LAYOUT);
	        tree.set_dirty_flags(&id, dirty);

	        let dirty = tree.dirty_flags(&id);
	        assert!(!dirty.contains(DirtyFlags::NEEDS_LAYOUT));
	        assert!(dirty.contains(DirtyFlags::NEEDS_PAINT));
    }

    #[test]
    fn ensure_preorder_indices_includes_detached_roots() {
        let mut tree = FiberTree::new();
        let root = tree.create_placeholder_fiber();
        let child = tree.create_placeholder_fiber();
        tree.relink_children_in_order(&root, &[child]);
        tree.root = Some(root);

        let detached = tree.create_placeholder_fiber();
        assert!(tree.parent(&detached).is_none());

        tree.ensure_preorder_indices();

        let mut indices = FxHashSet::default();
        for id in [root, child, detached] {
            let index = tree.preorder_index(&id).expect("missing preorder index");
            indices.insert(index);
        }
        assert_eq!(
            indices.len(),
            3,
            "all roots should get unique preorder indices"
        );
    }

    #[test]
    fn scene_segment_order_appends_deferred_by_priority() {
        let mut tree = FiberTree::new();
        let root = tree.create_placeholder_fiber();
        let child = tree.create_placeholder_fiber();
        let deferred_low = tree.create_placeholder_fiber();
        let deferred_high = tree.create_placeholder_fiber();
        tree.relink_children_in_order(&root, &[child, deferred_high, deferred_low]);
        tree.root = Some(root);

        let mut pool = crate::scene::SceneSegmentPool::default();
        let root_before = pool.alloc_segment();
        let child_before = pool.alloc_segment();
        let deferred_low_before = pool.alloc_segment();
        let deferred_high_before = pool.alloc_segment();

        tree.insert_scene_segments(
            &root,
            FiberSceneSegments {
                before: root_before,
                after: None,
            },
        );
        tree.insert_scene_segments(
            &child,
            FiberSceneSegments {
                before: child_before,
                after: None,
            },
        );
        tree.insert_scene_segments(
            &deferred_low,
            FiberSceneSegments {
                before: deferred_low_before,
                after: None,
            },
        );
        tree.insert_scene_segments(
            &deferred_high,
            FiberSceneSegments {
                before: deferred_high_before,
                after: None,
            },
        );

        tree.deferred_priorities.insert(deferred_low.into(), 0);
        tree.deferred_priorities.insert(deferred_high.into(), 5);

        let order = tree.scene_segment_order(root);
        let root_pos = order.iter().position(|id| *id == root_before).unwrap();
        let child_pos = order.iter().position(|id| *id == child_before).unwrap();
        let low_pos = order
            .iter()
            .position(|id| *id == deferred_low_before)
            .unwrap();
        let high_pos = order
            .iter()
            .position(|id| *id == deferred_high_before)
            .unwrap();

        assert!(child_pos > root_pos);
        assert!(low_pos > child_pos && high_pos > child_pos);
        assert!(low_pos < high_pos);
    }

    #[test]
    fn scene_segment_order_does_not_duplicate_cached_subtree_lists() {
        let mut tree = FiberTree::new();
        let root = tree.create_placeholder_fiber();
        let child = tree.create_placeholder_fiber();
        let grandchild = tree.create_placeholder_fiber();
        tree.relink_children_in_order(&root, &[child]);
        tree.relink_children_in_order(&child, &[grandchild]);
        tree.root = Some(root);

        let mut pool = crate::scene::SceneSegmentPool::default();
        let root_before = pool.alloc_segment();
        let child_before = pool.alloc_segment();
        let grandchild_before = pool.alloc_segment();

        tree.insert_scene_segments(
            &root,
            FiberSceneSegments {
                before: root_before,
                after: None,
            },
        );
        tree.insert_scene_segments(
            &child,
            FiberSceneSegments {
                before: child_before,
                after: None,
            },
        );
        tree.insert_scene_segments(
            &grandchild,
            FiberSceneSegments {
                before: grandchild_before,
                after: None,
            },
        );

        // Simulate cached subtree segment lists as produced by paint traversal.
        tree.paint_cache
            .get_mut(root.into())
            .unwrap()
            .scene_segment_list = Some(vec![root_before, child_before, grandchild_before]);
        tree.paint_cache
            .get_mut(child.into())
            .unwrap()
            .scene_segment_list = Some(vec![child_before, grandchild_before]);
        tree.paint_cache
            .get_mut(grandchild.into())
            .unwrap()
            .scene_segment_list = Some(vec![grandchild_before]);

        let order = tree.scene_segment_order(root);
        assert_eq!(order, vec![root_before, child_before, grandchild_before]);
    }

    #[test]
    fn can_replay_prepaint_is_blocked_by_bounds_changes() {
        let mut tree = FiberTree::new();
        let (root, child, _grandchild) = build_chain(&mut tree);

        // Simulate an otherwise-clean cached subtree.
        for id in [root, child] {
            let empty_range =
                crate::LineLayoutIndex::default()..crate::LineLayoutIndex::default();
            tree.set_dirty_flags(&id, DirtyFlags::NONE);
            let cache = tree.paint_cache.get_mut(id.into()).unwrap();
            cache.prepaint_state = Some(super::PrepaintState {
                accessed_entities: FxHashSet::default(),
                line_layout_range: empty_range.clone(),
            });
            cache.paint_list = Some(super::PaintList {
                line_layout_range: empty_range,
            });
        }

        // A bounds change (e.g. from layout) should invalidate prepaint for the moved fiber and its ancestors.
        tree.mark_dirty(&child, DirtyFlags::NEEDS_PAINT);
        assert!(
            !tree.can_replay_prepaint(&child),
            "bounds-changed fibers must re-run prepaint so bounds-dependent state updates"
        );
        assert!(
            !tree.can_replay_prepaint(&root),
            "ancestors with paint-dirty descendants must not replay prepaint"
        );
    }

    #[test]
    fn layout_state_is_stored_separately() {
        let mut tree = FiberTree::new();
        let id = tree.create_placeholder_fiber();
        let key = id.into();
        assert!(tree.layout_state.get(key).is_some());

        tree.remove(&id);
        assert!(tree.layout_state.get(key).is_none());
    }

    #[test]
    fn paint_cache_is_stored_separately() {
        let mut tree = FiberTree::new();
        let id = tree.create_placeholder_fiber();
        let key = id.into();
        assert!(tree.paint_cache.get(key).is_some());

        tree.remove(&id);
        assert!(tree.paint_cache.get(key).is_none());
    }

    #[test]
    fn view_state_is_stored_separately() {
        let mut tree = FiberTree::new();
        let id = tree.create_placeholder_fiber();
        let key = id.into();
        assert!(tree.view_state.get(key).is_some());

        tree.remove(&id);
        assert!(tree.view_state.get(key).is_none());
    }

    #[test]
    fn hitbox_state_is_stored_separately() {
        let mut tree = FiberTree::new();
        let id = tree.create_placeholder_fiber();
        let key = id.into();
        assert!(tree.hitbox_state.get(key).is_some());

        tree.remove(&id);
        assert!(tree.hitbox_state.get(key).is_none());
    }
