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
        /// The ID of the dummy beta node at the root of the network.
        dummy_node_id: usize,
        /// The ID of the dummy token stored in the dummy beta node.
        dummy_token_id: usize,
    },
    /// A WME was inserted into working memory.
    #[serde(rename_all = "camelCase")]
    AddedWme {
        /// The timetag of the new WME.
        timetag: usize,
        /// The symbol ID of the WME's ID.
        id: usize,
        /// The symbol ID of the WME's attribute.
        attribute: usize,
        /// The symbol ID of the WME's value.
        value: usize,
    },
    /// A WME was removed from working memory.
    #[serde(rename_all = "camelCase")]
    RemovedWme {
        /// The timetag of the removed WME.
        id: usize,
    },
    /// A production was added to the rete.
    #[serde(rename_all = "camelCase")]
    AddedProduction {
        /// The ID of the new production.
        id: usize,
        /// The ID of the production's node in the beta network.
        p_node_id: usize,
    },
    /// A production was removed from the rete.
    #[serde(rename_all = "camelCase")]
    RemovedProduction {
        /// The ID of the production that was removed.
        id: usize,
    },
    /// A token was created in a beta node.
    #[serde(rename_all = "camelCase")]
    AddedToken {
        /// The ID of the new token.
        id: usize,
        /// The ID of the beta node in which the new token is stored.
        node_id: usize,
        /// The ID of the token for the previous partial match.
        /// Following these willeventually lead to the dummy token.
        parent_id: usize,
    },
    /// A token was removed from a beta node.
    #[serde(rename_all = "camelCase")]
    RemovedToken {
        /// The ID of the token that was removed.
        id: usize,
    },
    /// A node was created in the rete network.
    #[serde(rename_all = "camelCase")]
    AddedNode {
        /// The ID of the node that was added.
        id: usize,
        /// The ID of the node's parent. Following these links will
        /// eventually lead to the dummy beta node.
        parent_id: usize,
        /// The type of node that was added.
        kind: NodeKind,
        /// The IDs of any child nodes.
        #[deprecated = "Nodes are never created with children already existing."]
        children: Vec<usize>,
        /// The ID of an alpha node that links to this node, or `None`
        /// if it doesn't have one. This is only set for join nodes.
        alpha_node_id: Option<usize>,
    },
    /// A node was removed from the rete network.
    #[serde(rename_all = "camelCase")]
    RemovedNode {
        /// The ID of the node that was removed.
        id: usize,
    },
    /// An alpha memory was created.
    #[serde(rename_all = "camelCase")]
    AddedAlphaMemory {
        /// The ID of the memory that was added.
        id: usize,
        /// The pattern that this alpha memory tests for.
        test: AlphaMemoryTest,
    },
    /// An alpha memory was removed.
    #[serde(rename_all = "camelCase")]
    RemovedAlphaMemory {
        /// The ID of the memory that was removed.
        id: usize,
    },
    /// A production was matched.
    #[serde(rename_all = "camelCase")]
    MatchedProduction {
        /// The ID of the production that matched.
        id: usize,
        /// The ID of the token that caused the match. Following its
        /// parents up to the root will reveal the list of WMEs that
        /// form the match.
        token: usize,
    },
    /// A production match was retracted.
    #[serde(rename_all = "camelCase")]
    UnmatchedProduction {
        /// The ID of the production that no longer matches.
        id: usize,
        /// The ID of the token that was removed and therefore caused
        /// the retraction.
        token: usize,
    },
}

/// The different kinds of nodes that can exist in the rete network.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub enum NodeKind {
    /// An alpha node.
    #[deprecated = "Alpha nodes have their own trace type"]
    Alpha,
    /// A beta node.
    Beta,
    /// A join node.
    Join,
    /// A production node.
    P,
}

/// The symbol IDs of the test that an alpha memory is performing.
///
/// For each field, `Some(id)` means that the test matches the symbol
/// corresponding to `id`, and `None` means that it matches a
/// wildcard.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct AlphaMemoryTest {
    /// The pattern being matched in the id position.
    pub id: Option<usize>,
    /// The pattern being matched in the attribute position.
    pub attribute: Option<usize>,
    /// The pattern being matched in the value position.
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

/// An implementation of `slog::Drain` that can capture `Trace`
/// events.
pub struct ObserverDrain<F: Fn(&Trace)> {
    callback: F,
}

impl<F: Fn(&Trace)> ObserverDrain<F> {
    /// Construct a new drain that will invoke the given callback when
    /// it receives messages.
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
