"""Undo and redo of bounding box edits.

The history is made of snapshots of everything that defines the boxes (corners,
features, colors, edge widths and z-indices). Undo and redo replace the whole
state of the layer, so they work the same for every kind of edit.

A history step is opened right before a user edit starts and closed when it is
completed: the snapshot taken at the start is pushed to the undo stack only if
the edit changed something. Several operations can be nested in one step, so a
whole mouse drag is one step, however many mouse moves or internal calls it
consists of. The current state is pushed to the opposite stack when undoing or
redoing, so changes made by other code after an edit (e.g. a plugin filling the
feature table of a box right after it was added) are kept by undo and redo.

Replacing ``data`` or ``features`` from code is not an edit of the user. It is
not recorded and it clears the history, so nothing recorded before can be
restored over the new state.
"""
import functools
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from napari.layers.base._base_constants import ActionType

from .._helper_functions import (
    layer_dims_not_displayed,
    layer_ndisplay,
    layer_slice_indices,
)
from .._utils import NAPARI_VERSION

MAX_HISTORY = 50
"""Maximum number of undo steps (and of redo steps) kept by a layer."""


@dataclass
class _Snapshot:
    """The state of a bounding box layer that undo and redo restore."""

    data: List[np.ndarray]
    edge_width: list
    z_index: list
    edge_color: np.ndarray
    face_color: np.ndarray
    features: pd.DataFrame

    def __len__(self):
        return len(self.data)

    def box_equals(self, other, index, other_index):
        """True if box `index` of this snapshot is the same as box `other_index` of `other`."""
        return (
            np.array_equal(self.data[index], other.data[other_index])
            and self.edge_width[index] == other.edge_width[other_index]
            and self.z_index[index] == other.z_index[other_index]
            and np.array_equal(self.edge_color[index], other.edge_color[other_index])
            and np.array_equal(self.face_color[index], other.face_color[other_index])
        )

    def equals(self, other):
        return (
            len(self) == len(other)
            and all(self.box_equals(other, i, i) for i in range(len(self)))
            and self.features.equals(other.features)
        )


def undoable(method):
    """Make a layer method one undo step (a no-op if it changes nothing)."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._history_step():
            return method(self, *args, **kwargs)

    return wrapper


def undoable_drag(callback):
    """Make a mouse drag callback one undo step, from the press to the release."""

    @functools.wraps(callback)
    def wrapper(layer, event):
        with layer._history_step():
            yield from callback(layer, event)

    return wrapper


def resets_history(method):
    """Mark a property setter that replaces a part of the state from code.

    The history is cleared if the replacement changes the state, and the
    replacement is not recorded. Such a change inside an edit (e.g. a drag) is
    not treated as a reset, the edit is recorded as usual.
    """

    @functools.wraps(method)
    def wrapper(self, value):
        outside_edit = not (self._history_step_open or self._history_suspended)
        before = (
            self._snapshot()
            if outside_edit and (self.can_undo or self.can_redo)
            else None
        )
        with self._suspend_history():
            method(self, value)
        if before is not None and not before.equals(self._snapshot()):
            self.clear_history()

    return wrapper


class UndoRedoMixin:
    """Undo and redo for bounding box layers (see the module docstring)."""

    def _init_history(self):
        self._undo_stack: List[_Snapshot] = []
        self._redo_stack: List[_Snapshot] = []
        self._step_start = None
        self._history_suspended = False

    @property
    def _history_step_open(self):
        return self._step_start is not None

    @property
    def can_undo(self) -> bool:
        """bool: True if there is an edit to undo."""
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        """bool: True if there is an undone edit to redo."""
        return len(self._redo_stack) > 0

    def clear_history(self):
        """Forget all the edits that can be undone or redone."""
        self._undo_stack.clear()
        self._redo_stack.clear()

    @contextmanager
    def _suspend_history(self):
        """Do not record the changes made inside the context."""
        previous = self._history_suspended
        self._history_suspended = True
        try:
            yield
        finally:
            self._history_suspended = previous

    @contextmanager
    def _history_step(self):
        """Record the changes made inside the context as one undo step.

        Only the outermost step is recorded, the inner ones are part of it.
        """
        if self._history_suspended or self._history_step_open:
            yield
            return
        start = self._step_start = self._snapshot()
        try:
            yield
        finally:
            self._step_start = None
            if not start.equals(self._snapshot()):
                self._undo_stack.append(start)
                del self._undo_stack[:-MAX_HISTORY]
                self._redo_stack.clear()

    def undo(self) -> bool:
        """Undo the last edit.

        Returns
        -------
        bool
            False if there was nothing to undo or an edit is in progress.
        """
        return self._travel(self._undo_stack, self._redo_stack)

    def redo(self) -> bool:
        """Redo the last undone edit.

        Returns
        -------
        bool
            False if there was nothing to redo or an edit is in progress.
        """
        return self._travel(self._redo_stack, self._undo_stack)

    def _travel(self, source, destination) -> bool:
        if not source or self._history_step_open or self._history_suspended:
            return False
        target = source.pop()
        current = self._snapshot()
        destination.append(current)
        del destination[:-MAX_HISTORY]
        self._restore(current, target)
        return True

    def _snapshot(self) -> _Snapshot:
        data_view = self._data_view
        return _Snapshot(
            data=[np.array(box.data) for box in data_view.bounding_boxes],
            edge_width=list(data_view.edge_widths),
            z_index=list(data_view.z_indices),
            edge_color=np.array(data_view._edge_color).reshape(-1, 4),
            face_color=np.array(data_view._face_color).reshape(-1, 4),
            features=self.features.copy(),
        )

    def _restore(self, current: _Snapshot, target: _Snapshot):
        """Replace the state of the layer, with the events of a normal edit."""
        removed, added, changed = _differences(current, target)
        self._emit_data_event(removed, added, changed, done=False)

        # nothing is being dragged any more, whatever it was
        self._finish_drawing()
        self._select_drag_active = False
        self._is_creating = False
        with self._suspend_history():
            # the features and text go first. The visuals must not draw the
            # text while it does not match the boxes, they update with the boxes
            defaults = self.feature_defaults
            self._feature_table.set_values(target.features, num_data=len(target))
            if list(defaults.columns) == list(self.feature_defaults.columns):
                self._feature_table.set_defaults(defaults)
            with self.text.events.blocker():
                self.text.refresh(self.features)
            data_view = type(self._data_view)(ndisplay=layer_ndisplay(self))
            data_view.slice_key = np.array(layer_slice_indices(self))[
                list(layer_dims_not_displayed(self))
            ]
            self._data_view = data_view
            self._add_bounding_boxes(
                target.data,
                edge_width=target.edge_width,
                edge_color=target.edge_color,
                face_color=target.face_color,
                z_index=target.z_index,
            )
        self.events.features()
        self.events.properties()
        self._update_dims()
        self._set_highlight(force=True)
        self._update_thumbnail()
        self.events.extent()

        self._emit_data_event(removed, added, changed, done=True)

    def _emit_data_event(self, removed, added, changed, done: bool):
        """Emit the data event of the edit that undo or redo is equal to.

        Both the events before and after the change are emitted, so a listener
        never sees an edit that is still in progress after undo or redo returned.
        """
        if removed:
            kind, indices = 'REMOVE', removed
        elif added:
            kind, indices = 'ADD', added
        else:
            kind, indices = 'CHANGE', changed
        if NAPARI_VERSION >= "0.4.19":
            names = {'ADD': ('ADDING', 'ADDED'), 'REMOVE': ('REMOVING', 'REMOVED'), 'CHANGE': ('CHANGING', 'CHANGED')}
            action = getattr(ActionType, names[kind][done])
        elif done:
            action = getattr(ActionType, kind).value
        else:
            return
        if kind == 'CHANGE':
            vertex_indices = tuple(
                tuple(range(len(self.data[i]))) for i in indices
            )
        else:
            vertex_indices = ((),)
        self.events.data(
            value=self.data,
            action=action,
            data_indices=tuple(indices),
            vertex_indices=vertex_indices,
        )


def _differences(before: _Snapshot, after: _Snapshot):
    """Indices of the boxes removed from `before`, added in `after` and changed.

    Boxes are matched in order, as an edit removes or adds boxes without
    reordering the others.
    """
    if len(after) == len(before):
        changed = [
            i
            for i in range(len(before))
            if not before.box_equals(after, i, i)
            or not before.features.iloc[i].equals(after.features.iloc[i])
        ]
        return [], [], changed or list(range(len(before)))
    smaller, larger = (after, before) if len(after) < len(before) else (before, after)
    missing = []
    j = 0
    for i in range(len(larger)):
        if j < len(smaller) and larger.box_equals(smaller, i, j):
            j += 1
        else:
            missing.append(i)
    if j < len(smaller):
        # not the result of removing or adding boxes, assume the last ones
        missing = list(range(len(smaller), len(larger)))
    if len(after) < len(before):
        return missing, [], []
    return [], missing, []
