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

pub struct Rete {
    alpha_tests: HashMap<AlphaTest, AlphaMemoryId>,
    alpha_network: HashMap<AlphaMemoryId, AlphaMemory>,
    beta_network: HashMap<ReteNodeId, ReteNode>,
    dummy_node_id: ReteNodeId,
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

impl Default for Rete {
    fn default() -> Self {
        Self::new()
    }
}

impl Rete {
    /// TODO: The Default impl is no longer correct because it doesn't
    /// create the initial beta node.
    pub fn new() -> Self {
        let id_generator = IdGenerator::default();

        // Since the parent is generated instead of acquired from
        // another node, it is a *dangling pointer*.
        let dummy_node = ReteNode {
            id: ReteNodeId(id_generator.next()),
            parent: ReteNodeId(id_generator.next()),
            children: vec![],
            kind: ReteNodeKind::Beta { tokens: vec![] },
        };
        let dummy_node_id = dummy_node.id;
        let mut beta_network = HashMap::new();
        beta_network.insert(dummy_node.id, dummy_node);

        Rete {
            alpha_tests: HashMap::new(),
            alpha_network: HashMap::new(),
            beta_network,
            dummy_node_id,
            tokens: HashMap::new(),
            productions: HashMap::new(),
            wme_alpha_memories: HashMap::new(),
            wme_tokens: HashMap::new(),
            token_children: HashMap::new(),
            pending_activations: Vec::new(),
            id_generator,
        }
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
        let mut current_node = self.dummy_node_id;

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

struct Condition([ConditionTest; 3]);

#[derive(Clone, Copy, PartialEq)]
struct VariableID(usize);

enum ConditionTest {
    Constant(SymbolID),
    Variable(VariableID),
}

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

        pub const ON: usize = 10;
        pub const COLOR: usize = 11;
        pub const LEFT_OF: usize = 12;

        pub const RED: usize = 20;
        pub const MAIZE: usize = 21;
        pub const GREEN: usize = 22;
        pub const BLUE: usize = 23;
        pub const WHITE: usize = 24;
        pub const TABLE: usize = 25;

        pub const W1: Wme = Wme([B1, ON, B2]);
        pub const W2: Wme = Wme([B1, ON, B3]);
        pub const W3: Wme = Wme([B1, COLOR, RED]);
        pub const W4: Wme = Wme([B2, ON, TABLE]);
        pub const W5: Wme = Wme([B2, LEFT_OF, B3]);
        pub const W6: Wme = Wme([B2, COLOR, BLUE]);
        pub const W7: Wme = Wme([B3, LEFT_OF, B4]);
        pub const W8: Wme = Wme([B3, ON, TABLE]);
        pub const W9: Wme = Wme([B3, COLOR, RED]);
        fn wmes() -> Vec<Wme> {
            vec![W1, W2, W3, W4, W5, W6, W7, W8, W9]
        }

        const V_X: ConditionTest = ConditionTest::Variable(VariableID(0));
        const V_Y: ConditionTest = ConditionTest::Variable(VariableID(1));
        const V_Z: ConditionTest = ConditionTest::Variable(VariableID(2));
        const V_A: ConditionTest = ConditionTest::Variable(VariableID(3));
        const V_B: ConditionTest = ConditionTest::Variable(VariableID(4));
        const V_C: ConditionTest = ConditionTest::Variable(VariableID(5));
        const V_D: ConditionTest = ConditionTest::Variable(VariableID(6));
        const V_S: ConditionTest = ConditionTest::Variable(VariableID(7));

        const C_ON: ConditionTest = ConditionTest::Constant(ON);
        const C_LEFT_OF: ConditionTest = ConditionTest::Constant(LEFT_OF);
        const C_COLOR: ConditionTest = ConditionTest::Constant(COLOR);
        const C_RED: ConditionTest = ConditionTest::Constant(RED);
        const C_MAIZE: ConditionTest = ConditionTest::Constant(MAIZE);
        const C_BLUE: ConditionTest = ConditionTest::Constant(BLUE);
        const C_GREEN: ConditionTest = ConditionTest::Constant(GREEN);
        const C_WHITE: ConditionTest = ConditionTest::Constant(WHITE);
        const C_TABLE: ConditionTest = ConditionTest::Constant(TABLE);

        const C1: Condition = Condition([V_X, C_ON, V_Y]);
        const C2: Condition = Condition([V_Y, C_LEFT_OF, V_Z]);
        const C3: Condition = Condition([V_Z, C_COLOR, C_RED]);
        const C4: Condition = Condition([V_A, C_COLOR, C_MAIZE]);
        const C5: Condition = Condition([V_B, C_COLOR, C_BLUE]);
        const C6: Condition = Condition([V_C, C_COLOR, C_GREEN]);
        const C7: Condition = Condition([V_D, C_COLOR, C_WHITE]);
        const C8: Condition = Condition([V_S, C_ON, C_TABLE]);
        const C9: Condition = Condition([V_Y, V_A, V_B]);
        const C10: Condition = Condition([V_A, C_LEFT_OF, V_D]);

        fn productions() -> Vec<Production> {
            vec![
                Production {
                    id: ProductionID(1),
                    conditions: vec![C1, C2, C3],
                },
                Production {
                    id: ProductionID(2),
                    conditions: vec![C1, C2, C4, C5],
                },
                Production {
                    id: ProductionID(3),
                    conditions: vec![C1, C2, C4, C3],
                },
            ]
        }

        #[test]
        fn add_productions() {
            let mut rete = Rete::default();
            for p in productions() {
                rete.add_production(p);
            }
        }

        #[test]
        fn add_productions_and_wmes() {
            let mut rete = Rete::default();
            for p in productions() {
                rete.add_production(p);
            }
            for wme in wmes() {
                rete.add_wme(wme);
            }

            unimplemented!("check matches")
        }

        #[test]
        fn add_wmes_then_productions() {
            let mut rete = Rete::default();
            for wme in wmes() {
                rete.add_wme(wme);
            }
            for p in productions() {
                rete.add_production(p);
            }

            unimplemented!("check matches")
        }
    }
}
