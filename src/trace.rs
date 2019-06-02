//! Tools for observing changes in the rete. Requires the `trace` feature.

use serde::{Deserialize, Serialize};
use slog::{Drain, Key, OwnedKVList, Record, SerdeValue, Serializer, KV};
use std::fmt;

/// A description of a modification that happened in the rete.
///
/// The rete will emit values of this type via its logging
/// infrastructure.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum Trace {
    /// Emitted once when a rete is constructed.
    Initialized,
    /// A WME was inserted into working memory.
    AddedWme { id: usize },
    /// A WME was removed from working memory.
    RemovedWme { id: usize },
    /// A production was added to the rete.
    AddedProduction { id: usize },
    /// A production was removed from the rete.
    RemovedProduction { id: usize },
    /// A token was created in a beta node.
    AddedToken {
        node_id: usize,
        token_id: usize,
        parent_token_id: usize,
    },
    /// A token was removed from a beta node.
    RemovedToken { token_id: usize },
    /// A node was created in the rete network.
    AddedNode {
        id: usize,
        parent_id: usize,
        kind: NodeKind,
    },
    /// A node was removed from the rete network.
    RemovedNode { id: usize },
}

/// The different kinds of nodes that can exist in the rete network.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub enum NodeKind {
    Alpha,
    Beta,
    Join,
    P,
}

impl slog::Value for Trace {
    fn serialize(&self, _: &Record, key: Key, serializer: &mut Serializer) -> slog::Result {
        serializer.emit_serde(key, self)
    }
}

impl SerdeValue for Trace {
    fn as_serde(&self) -> &erased_serde::Serialize {
        self
    }

    fn to_sendable(&self) -> Box<SerdeValue + Send + 'static> {
        Box::new(self.clone())
    }

    fn serialize_fallback(&self, key: Key, serializer: &mut Serializer) -> slog::Result {
        serializer.emit_str(key, "<< EVENT >>")
    }
}

impl KV for Trace {
    fn serialize(&self, _: &Record, serializer: &mut Serializer) -> slog::Result {
        serializer.emit_serde("event", self)
    }
}

pub struct ObserverDrain;

impl Drain for ObserverDrain {
    type Ok = ();
    type Err = slog::Error;

    fn log(&self, record: &Record, _: &OwnedKVList) -> Result<Self::Ok, Self::Err> {
        struct S;
        impl Serializer for S {
            fn emit_arguments(&mut self, _: Key, _: &fmt::Arguments) -> slog::Result {
                Ok(())
            }

            fn emit_serde(&mut self, _: Key, value: &SerdeValue) -> slog::Result {
                let buffer = serde_json::to_string(value.as_serde()).unwrap();
                if let Ok(event) = serde_json::from_str::<Trace>(&buffer) {
                    println!("{:?}", event);
                }
                Ok(())
            }
        }
        record.kv().serialize(record, &mut S)
    }
}
