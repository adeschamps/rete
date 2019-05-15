use std::{
    collections::HashMap,
    sync::atomic::{AtomicUsize, Ordering},
};

type SymbolID = usize;

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct Wme([SymbolID; 3]);

/// Generates unique identifiers.
#[derive(Default)]
struct IdGenerator(AtomicUsize);

impl IdGenerator {
    /// Get the next id. This will be different every time this
    /// function is called.
    fn next(&self) -> usize {
        self.0.fetch_add(1, Ordering::Relaxed)
    }
}

#[derive(Default)]
pub struct Rete {
    alpha_tests: HashMap<AlphaTest, AlphaMemoryId>,
    alpha_network: HashMap<AlphaMemoryId, AlphaMemory>,
    beta_network: HashMap<ReteNodeId, ReteNode>,
    tokens: HashMap<TokenId, Token>,

    productions: HashMap<ProductionID, Production>,

    /// Whenever a WME is added to an alpha memory, the alpha memory
    /// is added to this collection. This makes removals more
    /// efficient.
    wme_alpha_memories: HashMap<Wme, Vec<AlphaMemoryId>>,
    /// Whenever a token is created, it is added to this collection.
    /// This makes removals more efficient.
    wme_tokens: HashMap<Wme, Vec<TokenId>>,

    /// For each token, the list of tokens that are its children.
    ///
    /// TODO: This could be replaced with a proper graph.
    token_children: HashMap<TokenId, Vec<TokenId>>,

    /// To avoid recursive function calls when activating memory
    /// nodes, we queue up activations here. The activate_memories
    /// function consumes these until the vector is empty.
    pending_activations: Vec<Activation>,

    id_generator: IdGenerator,
}

enum Activation {
    BetaLeft(ReteNodeId, TokenId),
    BetaRight(ReteNodeId, Wme),
}

impl Rete {
    /// TODO: The Default impl is no longer correct because it doesn't
    /// create the initial beta node.
    fn new() -> Self {
        let mut rete = Self::default();

        // Since the parent is generated instead of acquired from
        // another node, it is a *dangling pointer*.
        let dummy_node = ReteNode {
            id: ReteNodeId(rete.id_generator.next()),
            parent: ReteNodeId(rete.id_generator.next()),
            children: vec![],
            kind: ReteNodeKind::Beta { tokens: vec![] },
        };
        rete.beta_network.insert(dummy_node.id, dummy_node);
        rete
    }

    pub fn add_wme(&mut self, wme: Wme) {
        let tests = [
            AlphaTest([None, None, None]),
            AlphaTest([None, None, Some(wme.0[2])]),
            AlphaTest([None, Some(wme.0[1]), None]),
            AlphaTest([None, Some(wme.0[1]), Some(wme.0[2])]),
            AlphaTest([Some(wme.0[0]), None, None]),
            AlphaTest([Some(wme.0[0]), None, Some(wme.0[2])]),
            AlphaTest([Some(wme.0[0]), Some(wme.0[1]), None]),
            AlphaTest([Some(wme.0[0]), Some(wme.0[1]), Some(wme.0[2])]),
        ];

        for test in &tests {
            if let Some(alpha_memory_id) = self.alpha_tests.get_mut(test) {
                let alpha_memory = self.alpha_network.get_mut(alpha_memory_id).unwrap();
                // Activate alpha memory
                alpha_memory.wmes.push(wme);
                self.wme_alpha_memories
                    .entry(wme)
                    .or_default()
                    .push(alpha_memory.id);
                for join_node_id in &alpha_memory.successors {
                    // right activate join node
                    self.pending_activations
                        .push(Activation::BetaRight(*join_node_id, wme));
                }
            }
        }

        self.activate_memories();
    }

    pub fn remove_wme(&mut self, wme: Wme) {
        let alpha_memories = self
            .wme_alpha_memories
            .remove(&wme)
            .expect("removing WME that is not in any alpha memories.");
        for memory in alpha_memories {
            let alpha_memory = self.alpha_network.get_mut(&memory).unwrap();
            alpha_memory.wmes.retain(|w| *w != wme);
        }

        let mut tokens_to_remove = self
            .wme_tokens
            .remove(&wme)
            .expect("removing WME that is not in any tokens.");
        while let Some(token) = tokens_to_remove.pop() {
            let token = self.tokens.remove(&token).unwrap();
            if let Some(more_tokens) = self.token_children.remove(&token.id) {
                tokens_to_remove.extend(more_tokens);
            }
            let beta_node = self.beta_network.get_mut(&token.node).unwrap();
            match beta_node.kind {
                ReteNodeKind::Beta { ref mut tokens } => tokens.retain(|t| *t != token.id),
                _ => unreachable!(),
            }
        }
    }

    pub fn add_production(&mut self, production: Production) {
        self.productions.insert(production.id, production);
        unimplemented!()
    }

    pub fn remove_production(&mut self, id: ProductionID) {
        self.productions.remove(&id);
        unimplemented!()
    }

    fn add_alpha_memory(&mut self, test: AlphaTest, alpha_memory: AlphaMemory) {
        self.alpha_tests.insert(test, alpha_memory.id);
        self.alpha_network.insert(alpha_memory.id, alpha_memory);
    }

    fn activate_memories(&mut self) {
        while let Some(activation) = self.pending_activations.pop() {
            let mut new_tokens = vec![];
            match activation {
                Activation::BetaLeft(beta_node_id, token) => {
                    let node = &self.beta_network[&beta_node_id];
                    match &node.kind {
                        ReteNodeKind::Beta { tokens } => unimplemented!(),
                        ReteNodeKind::Join {
                            alpha_memory,
                            tests,
                        } => {
                            let mut new_activations = vec![];
                            let alpha_memory = &self.alpha_network[alpha_memory];
                            let token = &self.tokens[&token];
                            for wme in alpha_memory.wmes.iter().filter(|wme| {
                                tests.iter().all(|test| self.join_test(test, token, wme))
                            }) {
                                let new_token = Token {
                                    id: TokenId(self.id_generator.next()),
                                    parent: token.id,
                                    node: node.id,
                                    wme: *wme,
                                };
                                new_tokens.push(new_token);
                                new_activations.extend(
                                    node.children
                                        .iter()
                                        .map(|id| Activation::BetaLeft(*id, new_token.id)),
                                );
                            }
                            self.pending_activations.extend(new_activations);
                        }
                        ReteNodeKind::P { .. } => unimplemented!(),
                    }
                }
                Activation::BetaRight(beta_node_id, wme) => {
                    let node = &self.beta_network[&beta_node_id];
                    match &node.kind {
                        ReteNodeKind::Beta { .. } => unimplemented!(),
                        ReteNodeKind::Join {
                            alpha_memory,
                            tests,
                        } => {
                            let beta_node = &self.beta_network[&node.parent];
                            let tokens = match beta_node.kind {
                                ReteNodeKind::Beta { ref tokens } => tokens,
                                _ => unreachable!(),
                            };
                            let new_activations: Vec<_> = tokens
                                .iter()
                                .filter_map(|token| self.tokens.get(token))
                                .filter(|token| {
                                    tests.iter().all(|test| self.join_test(test, token, &wme))
                                })
                                .flat_map(|token| {
                                    node.children
                                        .iter()
                                        .map(move |id| Activation::BetaLeft(*id, token.id))
                                })
                                .collect();
                            self.pending_activations.extend(new_activations);
                        }
                        ReteNodeKind::P { .. } => unimplemented!(),
                    }
                }
            }
            for new_token in new_tokens {
                self.tokens.insert(new_token.id, new_token);
            }
        }
    }

    // Generate a new token and return its id.
    fn make_token(&self, node: &ReteNode, parent: &Token, wme: Wme) -> TokenId {
        let id = TokenId(self.id_generator.next());
        let token = Token {
            id,
            parent: parent.id,
            node: node.id,
            wme,
        };
        // self.tokens.insert(id, token);
        id
    }

    fn join_test(&self, test: &JoinNodeTest, token: &Token, wme: &Wme) -> bool {
        let mut token = token;
        for _ in 0..test.beta_condition_offset {
            token = &self.tokens[&token.parent];
        }
        let other_wme = token.wme;

        wme.0[test.alpha_field] == other_wme.0[test.beta_field]
    }
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct AlphaTest([Option<SymbolID>; 3]);

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct AlphaMemoryId(usize);

struct AlphaMemory {
    id: AlphaMemoryId,
    wmes: Vec<Wme>,
    successors: Vec<ReteNodeId>,
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct ReteNodeId(usize);

struct ReteNode {
    id: ReteNodeId,
    children: Vec<ReteNodeId>,
    parent: ReteNodeId,
    kind: ReteNodeKind,
}

enum ReteNodeKind {
    Beta {
        tokens: Vec<TokenId>,
    },
    Join {
        alpha_memory: AlphaMemoryId,
        tests: Vec<JoinNodeTest>,
    },
    P {
        production: ProductionID,
    },
}

struct JoinNodeTest {
    // These two should be able to fit in a single byte.
    alpha_field: usize,
    beta_field: usize,

    beta_condition_offset: usize,
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct TokenId(usize);

#[derive(Clone, Copy)]
struct Token {
    id: TokenId,
    parent: TokenId,
    wme: Wme,
    /// The rete node containing this token.
    node: ReteNodeId,
}

//////////

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct ProductionID(usize);

pub struct Production {
    id: ProductionID,
    conditions: Vec<Condition>,
}

struct Condition;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_rete() {
        let rete = Rete::new();
        assert_eq!(rete.alpha_network.len(), 0);
        assert_eq!(rete.beta_network.len(), 1); // dummy top node.
    }

    #[test]
    fn add_wmes() {
        let mut rete = Rete::new();
        let test = AlphaTest([Some(0), Some(1), None]);
        let alpha_memory = AlphaMemory {
            id: AlphaMemoryId(0),
            wmes: vec![],
            successors: vec![],
        };
        rete.add_alpha_memory(test, alpha_memory);
        let test = AlphaTest([Some(99), None, None]);
        let alpha_memory = AlphaMemory {
            id: AlphaMemoryId(1),
            wmes: vec![],
            successors: vec![],
        };
        rete.add_alpha_memory(test, alpha_memory);

        assert_eq!(rete.alpha_network.len(), 2);

        let wme = Wme([0, 1, 2]);
        rete.add_wme(wme);

        let alpha_memory = &rete.alpha_network[&AlphaMemoryId(0)];
        assert_eq!(alpha_memory.wmes.len(), 1);

        let alpha_memory = &rete.alpha_network[&AlphaMemoryId(1)];
        assert_eq!(alpha_memory.wmes.len(), 0);
    }

    mod blocks {
        use crate::*;

        pub const B1: usize = 1;
        pub const B2: usize = 2;
        pub const B3: usize = 3;
        pub const B4: usize = 4;

        pub const on: usize = 10;
        pub const color: usize = 11;
        pub const left_of: usize = 12;

        pub const red: usize = 20;
        pub const maize: usize = 21;
        pub const green: usize = 22;
        pub const blue: usize = 23;
        pub const white: usize = 24;
        pub const table: usize = 25;

        pub const w1: Wme = Wme([B1, on, B2]);
        pub const w2: Wme = Wme([B1, on, B3]);
        pub const w3: Wme = Wme([B1, color, red]);
        pub const w4: Wme = Wme([B2, on, table]);
        pub const w5: Wme = Wme([B2, left_of, B3]);
        pub const w6: Wme = Wme([B2, color, blue]);
        pub const w7: Wme = Wme([B3, left_of, B4]);
        pub const w8: Wme = Wme([B3, on, table]);
        pub const w9: Wme = Wme([B3, color, red]);
        fn wmes() -> Vec<Wme> {
            vec![w1, w2, w3, w4, w5, w6, w7, w8, w9]
        }

        const t1: AlphaTest = AlphaTest([None, Some(on), None]);
        const t2: AlphaTest = AlphaTest([None, Some(left_of), None]);
        const t3: AlphaTest = AlphaTest([None, Some(color), Some(red)]);
        const t4: AlphaTest = AlphaTest([None, Some(color), Some(maize)]);
        const t5: AlphaTest = AlphaTest([None, Some(color), Some(blue)]);
        const t6: AlphaTest = AlphaTest([None, Some(color), Some(green)]);
        const t7: AlphaTest = AlphaTest([None, Some(color), Some(white)]);
        const t8: AlphaTest = AlphaTest([None, Some(on), Some(table)]);
        const t9: AlphaTest = AlphaTest([None, None, None]);
        const t10: AlphaTest = AlphaTest([None, Some(left_of), None]);
    }
}
