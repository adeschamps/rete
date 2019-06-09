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
    #[serde(rename_all = "camelCase")]
    Initialized {
        dummy_node_id: usize,
        dummy_token_id: usize,
    },
    /// A WME was inserted into working memory.
    #[serde(rename_all = "camelCase")]
    AddedWme {
        timetag: usize,
        id: usize,
        attribute: usize,
        value: usize,
    },
    /// A WME was removed from working memory.
    #[serde(rename_all = "camelCase")]
    RemovedWme { id: usize },
    /// A production was added to the rete.
    #[serde(rename_all = "camelCase")]
    AddedProduction { id: usize, p_node_id: usize },
    /// A production was removed from the rete.
    #[serde(rename_all = "camelCase")]
    RemovedProduction { id: usize },
    /// A token was created in a beta node.
    #[serde(rename_all = "camelCase")]
    AddedToken {
        node_id: usize,
        token_id: usize,
        parent_token_id: usize,
    },
    /// A token was removed from a beta node.
    #[serde(rename_all = "camelCase")]
    RemovedToken { token_id: usize },
    /// A node was created in the rete network.
    #[serde(rename_all = "camelCase")]
    AddedNode {
        id: usize,
        parent_id: usize,
        kind: NodeKind,
        // TODO: Is this needed?
        children: Vec<usize>,
        alpha_node_id: Option<usize>,
    },
    /// A node was removed from the rete network.
    #[serde(rename_all = "camelCase")]
    RemovedNode { id: usize },
    /// An alpha memory was created.
    #[serde(rename_all = "camelCase")]
    AddedAlphaMemory { id: usize, test: AlphaMemoryTest },
    /// An alpha memory was removed.
    #[serde(rename_all = "camelCase")]
    RemovedAlphaMemory { id: usize },
}

/// The different kinds of nodes that can exist in the rete network.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub enum NodeKind {
    Alpha,
    Beta,
    Join,
    P,
}

/// The symbol IDs of the test that an alpha memory is performing.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct AlphaMemoryTest {
    pub id: Option<usize>,
    pub attribute: Option<usize>,
    pub value: Option<usize>,
}

impl Trace {
    /// Traces are logged under this key. The drain will unsafely cast
    /// a `&SerdeValue` back to a `&Trace` if it matches this key, so
    /// **this key should not be used to log anything else**.
    const KEY: &'static str = concat!(
        env!("CARGO_PKG_NAME"),
        "_",
        env!("CARGO_PKG_VERSION"),
        "_unsafe_trace_key"
    );
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
        serializer.emit_serde(Trace::KEY, self)
    }
}

pub struct ObserverDrain<F: Fn(&Trace)> {
    callback: F,
}

impl<F: Fn(&Trace)> ObserverDrain<F> {
    pub fn new(callback: F) -> Self {
        Self { callback }
    }
}

impl<F: Fn(&Trace)> Drain for ObserverDrain<F> {
    type Ok = ();
    type Err = slog::Error;

    fn log(&self, record: &Record, _: &OwnedKVList) -> Result<Self::Ok, Self::Err> {
        struct S<'a, F>(&'a F);
        impl<'a, F: Fn(&Trace)> Serializer for S<'a, F> {
            fn emit_arguments(&mut self, _: Key, _: &fmt::Arguments) -> slog::Result {
                Ok(())
            }

            fn emit_serde(&mut self, key: Key, value: &SerdeValue) -> slog::Result {
                // This is only safe if we are sure that `value` is
                // definitely an instance of `Trace`. We attempt to
                // guarentee that be ensuring that the key is not used
                // by any other types. Thus, the safety of this cast
                // depends on having no name clashes with other keys.
                if key == Trace::KEY {
                    let event: &Trace = unsafe { &*(value as *const SerdeValue as *const Trace) };
                    self.0(event);
                }

                Ok(())
            }
        }
        record.kv().serialize(record, &mut S(&self.callback))
    }
}
