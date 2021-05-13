//! Tools for observing changes in the rete. Requires the `trace` feature.

use serde::{Deserialize, Serialize};

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
        /// The symbol ID of the WME's ID.
        id: usize,
        /// The symbol ID of the WME's attribute.
        attribute: usize,
        /// The symbol ID of the WME's value.
        value: usize,
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
