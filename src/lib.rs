#[macro_use]
extern crate slog;

use petgraph::{
    stable_graph::{NodeIndex, StableGraph},
    Directed, Direction,
};
use slog::{Drain, Logger};
use std::{
    collections::HashMap,
    sync::atomic::{AtomicUsize, Ordering},
};

type SymbolID = usize;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
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
    log: Logger,
    alpha_tests: HashMap<AlphaTest, AlphaMemoryId>,
    alpha_network: HashMap<AlphaMemoryId, AlphaMemory>,
    beta_network: StableGraph<ReteNode, (), Directed>,
    dummy_node_id: ReteNodeId,
    tokens: StableGraph<Token, (), Directed>,

    productions: HashMap<ProductionID, Production>,

    /// Whenever a WME is added to an alpha memory, the alpha memory
    /// is added to this collection. This makes removals more
    /// efficient.
    wme_alpha_memories: HashMap<Wme, Vec<AlphaMemoryId>>,
    /// Whenever a token is created, it is added to this collection.
    /// This makes removals more efficient.
    wme_tokens: HashMap<Wme, Vec<TokenId>>,

    /// To avoid recursive function calls when activating memory
    /// nodes, we queue up activations here. The activate_memories
    /// function consumes these until the vector is empty.
    pending_activations: Vec<Activation>,

    /// Events that have occurred since the last time this list was
    /// cleared.
    events: Vec<Event>,

    id_generator: IdGenerator,
}

#[derive(Debug, PartialEq)]
pub enum Event {
    Fired(ProductionID),
    Retracted(ProductionID),
}

#[derive(Debug)]
struct Activation {
    node: ReteNodeId,
    kind: ActivationKind,
}

#[derive(Debug)]
enum ActivationKind {
    Left(Wme, TokenId),
    Right(Wme),
}

impl Default for Rete {
    fn default() -> Self {
        Self::new(None)
    }
}

impl Rete {
    pub fn new(log: impl Into<Option<slog::Logger>>) -> Self {
        let log = log.into().unwrap_or_else(|| {
            let drain = slog_stdlog::StdLog;
            Logger::root(drain.fuse(), o!())
        });
        info!(log, "constructing rete");

        let id_generator = IdGenerator::default();

        let mut beta_network = StableGraph::new();
        let mut tokens = StableGraph::new();

        let dummy_node = ReteNode::Beta { tokens: vec![] };
        let dummy_node_id = beta_network.add_node(dummy_node);
        let dummy_token = Token {
            wme: Wme([0, 0, 0]),
            node: dummy_node_id,
        };
        let dummy_token_id = tokens.add_node(dummy_token);
        match &mut beta_network[dummy_node_id] {
            ReteNode::Beta { tokens } => tokens.push(dummy_token_id),
            _ => unreachable!("this is definitely a beta node"),
        }

        Rete {
            log,
            alpha_tests: HashMap::new(),
            alpha_network: HashMap::new(),
            beta_network,
            dummy_node_id,
            tokens,
            productions: HashMap::new(),
            wme_alpha_memories: HashMap::new(),
            wme_tokens: HashMap::new(),
            pending_activations: Vec::new(),
            events: Vec::new(),
            id_generator,
        }
    }

    pub fn add_wme(&mut self, wme: Wme) {
        let log = self.log.new(o!("wme" => format!("{:?}", wme)));
        trace!(log, "add wme");

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

        // Ensure we have an entry for this WME, even if no alpha
        // memories contain it.
        self.wme_alpha_memories.entry(wme).or_default();

        for test in &tests {
            let alpha_memory_id = match self.alpha_tests.get(test) {
                Some(id) => id,
                None => continue,
            };
            let alpha_memory = self.alpha_network.get_mut(alpha_memory_id).unwrap();

            let log = log.new(o!("alpha" => alpha_memory.id.0));
            trace!(log, "matched");

            // Activate alpha memory
            alpha_memory.wmes.push(wme);
            self.wme_alpha_memories
                .entry(wme)
                .or_default()
                .push(alpha_memory.id);
            for join_node_id in &alpha_memory.successors {
                // right activate join node
                self.pending_activations.push(Activation {
                    node: *join_node_id,
                    kind: ActivationKind::Right(wme),
                });
            }
        }

        self.activate_memories(log);
    }

    pub fn remove_wme(&mut self, wme: Wme) {
        let log = self.log.new(o!("wme" => format!("{:?}", wme)));
        trace!(log, "remove wme");

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
        while let Some(token_id) = tokens_to_remove.pop() {
            tokens_to_remove.extend(self.tokens.neighbors(token_id));
            let token = self.tokens.remove_node(token_id).unwrap();
            match self.beta_network[token.node] {
                ReteNode::Beta { ref mut tokens } => tokens.retain(|t| *t != token_id),
                _ => unreachable!(),
            }
        }
    }

    pub fn add_production(&mut self, production: Production) {
        let log = self.log.new(o!("production_id" => production.id.0));
        trace!(log, "add production"; "production" => ?production);

        let mut current_node_id = self.dummy_node_id;

        for i in 0..production.conditions.len() {
            let condition = production.conditions[i];
            trace!(log, "add condition"; "condition" => ?condition);

            // get join tests from condition
            // NOTE: This does not handle intra-condition tests.
            let tests: Vec<JoinNodeTest> = condition
                .variables()
                .filter_map(|(alpha_field, variable_id)| {
                    production.conditions[0..i]
                        .iter()
                        .rev()
                        .enumerate()
                        .find_map(|(distance, prev_condition)| {
                            prev_condition
                                .variables()
                                .find(|(_, var)| *var == variable_id)
                                .map(|(i, _)| (i, distance))
                        })
                        .map(|(beta_field, beta_condition_offset)| JoinNodeTest {
                            alpha_field,
                            beta_field,
                            beta_condition_offset,
                        })
                })
                .collect();

            // build or share alpha memory
            let alpha_test = AlphaTest::from(condition);
            // Returns either an existing id or generates a new one.
            let alpha_memory_id = self
                .alpha_tests
                .get(&alpha_test)
                .cloned()
                .unwrap_or_else(|| AlphaMemoryId(self.id_generator.next()));
            // If we need a new alpha memory, create it.
            if !self.alpha_network.contains_key(&alpha_memory_id) {
                // Collect the WMEs that should already be in this
                // memory.
                let wmes = self
                    .wme_alpha_memories
                    .iter_mut()
                    .filter(|(wme, _)| alpha_test.matches(wme))
                    .map(|(wme, alpha_memories)| {
                        // Keep track of which alpha memories contain
                        // this WME.
                        alpha_memories.push(alpha_memory_id);
                        *wme
                    })
                    .collect();

                let memory = AlphaMemory {
                    id: alpha_memory_id,
                    wmes,
                    successors: vec![],
                };
                trace!(log, "created alpha memory";
                    "test" => ?alpha_test,
                    "id" => ?memory.id,
                    "wmes" => memory.wmes.len());

                self.alpha_tests.insert(alpha_test, memory.id);
                self.alpha_network.insert(memory.id, memory);
            }
            let alpha_memory = &self.alpha_network[&alpha_memory_id];

            // build or share join node
            current_node_id = self
                .beta_network
                .neighbors(current_node_id)
                .find(|id| match self.beta_network[*id] {
                    ReteNode::Join {
                        alpha_memory: join_amem,
                        tests: ref join_tests,
                    } if join_amem == alpha_memory.id && *join_tests == tests => true,
                    _ => false,
                })
                .unwrap_or_else(|| {
                    // If we couldn't find a node to share, create
                    // one.
                    let new_node = ReteNode::Join {
                        alpha_memory: alpha_memory_id,
                        tests,
                    };
                    let id = self.beta_network.add_node(new_node);
                    self.beta_network.add_edge(current_node_id, id, ());
                    trace!(log, "create join node"; "id" => ?id);
                    // link to alpha memory
                    self.alpha_network
                        .get_mut(&alpha_memory_id)
                        .unwrap()
                        .successors
                        .push(id);
                    id
                });

            // build or share beta memory
            current_node_id = if i + 1 < production.conditions.len() {
                self.beta_network
                    .neighbors(current_node_id)
                    .find(|id| match self.beta_network[*id] {
                        ReteNode::Beta { .. } => true,
                        _ => false,
                    })
                    .unwrap_or_else(|| {
                        // If we couldn't find a node to share, create
                        // one.
                        let new_node = ReteNode::Beta { tokens: vec![] };
                        let id = self.beta_network.add_node(new_node);
                        self.beta_network.add_edge(current_node_id, id, ());
                        trace!(log, "create beta memory"; "id" => ?id);

                        self.activate_new_node(log.clone(), current_node_id, id);

                        id
                    })
            } else {
                current_node_id
            };
        }

        // Build new production node
        let new_node = ReteNode::P {
            production: production.id,
            activations: vec![],
        };
        let id = self.beta_network.add_node(new_node);
        trace!(log, "create p node"; "id" => ?id);
        self.beta_network.add_edge(current_node_id, id, ());
        self.activate_new_node(log.clone(), current_node_id, id);
        trace!(log, "new p node"; "id" => ?id);
    }

    pub fn remove_production(&mut self, id: ProductionID) {
        let log = self.log.clone();
        trace!(log, "remove production: {:?}", id);

        self.productions.remove(&id);
        unimplemented!()
    }

    pub fn take_events(&mut self) -> Vec<Event> {
        let mut events = Vec::new();
        std::mem::swap(&mut events, &mut self.events);
        events
    }

    fn activate_new_node(&mut self, log: Logger, parent: NodeIndex, new_node: NodeIndex) {
        let log = log.new(o!("new node" => new_node.index()));
        trace!(log, "activate new node");
        match self.beta_network[parent] {
            ReteNode::Beta { .. } => unimplemented!(),

            ReteNode::Join { alpha_memory, .. } => {
                trace!(log, "parent is a join node");
                let children: Vec<_> = self.beta_network.neighbors(parent).collect();
                for child in &children {
                    if let Some(edge) = self.beta_network.find_edge(parent, *child) {
                        self.beta_network.remove_edge(edge);
                    }
                }
                self.beta_network.add_edge(parent, new_node, ());
                for wme in &self.alpha_network[&alpha_memory].wmes {
                    self.pending_activations.push(Activation {
                        node: parent,
                        kind: ActivationKind::Right(*wme),
                    });
                }
                self.activate_memories(log);
                for child in children {
                    self.beta_network.update_edge(parent, child, ());
                }
            }

            ReteNode::P { .. } => unreachable!("P nodes never have any children"),
        }
    }

    fn activate_memories(&mut self, log: Logger) {
        trace!(log, "activate memories");
        while let Some(activation) = self.pending_activations.pop() {
            let log = log.new(o!("activation" => format!("{:?}", activation), "remaining" => self.pending_activations.len()));
            let node = &self.beta_network[activation.node];
            trace!(log, "activating");
            let mut new_tokens = vec![];
            let mut activations = vec![];
            match (&activation.kind, &node) {
                (ActivationKind::Left(wme, token), ReteNode::Beta { .. }) => {
                    let new_token = Token {
                        wme: *wme,
                        node: activation.node,
                    };
                    trace!(log, "new token"; "token" => ?new_token);
                    let new_token_id = self.tokens.add_node(new_token);
                    self.tokens.add_edge(new_token_id, *token, ());
                    new_tokens.push(new_token_id);
                    let new_activations: Vec<_> = self
                        .beta_network
                        .neighbors(activation.node)
                        .map(|id| Activation {
                            node: id,
                            kind: ActivationKind::Left(*wme, new_token_id),
                        })
                        .collect();
                    trace!(log, "enqueing {} new activations", new_activations.len());
                    self.pending_activations.extend(new_activations);
                }
                (
                    ActivationKind::Left(_, token),
                    ReteNode::Join {
                        alpha_memory,
                        tests,
                    },
                ) => {
                    let alpha_memory = &self.alpha_network[&alpha_memory];
                    let new_activations: Vec<_> = alpha_memory
                        .wmes
                        .iter()
                        .filter(|wme| tests.iter().all(|test| self.join_test(test, *token, wme)))
                        .flat_map(|wme| {
                            self.beta_network
                                .neighbors(activation.node)
                                .map(move |id| Activation {
                                    node: id,
                                    kind: ActivationKind::Left(*wme, *token),
                                })
                        })
                        .collect();

                    trace!(log, "enqueing {} new activations", new_activations.len());
                    self.pending_activations.extend(new_activations);
                }
                (ActivationKind::Left(_, token), ReteNode::P { production, .. }) => {
                    info!(log, "Activated P node"; "production" => ?production);
                    self.events.push(Event::Fired(*production));
                    activations.push((activation.node, token));
                }
                (ActivationKind::Right(_), ReteNode::Beta { .. }) => {
                    unreachable!("beta nodes are never right activated")
                }
                (
                    ActivationKind::Right(wme),
                    ReteNode::Join {
                        alpha_memory,
                        tests,
                    },
                ) => {
                    let beta_node = self
                        .beta_network
                        .neighbors_directed(activation.node, Direction::Incoming)
                        .next()
                        .unwrap();
                    let log = log.new(o!("alpha" => alpha_memory.0, "beta" => beta_node.index()));
                    let tokens = match self.beta_network[beta_node] {
                        ReteNode::Beta { ref tokens } => tokens,
                        _ => unreachable!("parent of a join node should be a beta node"),
                    };
                    trace!(log, "scanning {} tokens", tokens.len());
                    let new_activations: Vec<_> = tokens
                        .iter()
                        .filter(|token| {
                            tests.iter().all(|test| self.join_test(test, **token, &wme))
                        })
                        .flat_map(|token| {
                            self.beta_network
                                .neighbors(activation.node)
                                .map(move |id| Activation {
                                    node: id,
                                    kind: ActivationKind::Left(*wme, *token),
                                })
                        })
                        .collect();

                    trace!(log, "enqueing {} new activations", new_activations.len());
                    self.pending_activations.extend(new_activations);
                }
                (ActivationKind::Right(_), ReteNode::P { .. }) => {
                    unreachable!("p-nodes are never right activated")
                }
            }

            for new_token in new_tokens {
                match &mut self.beta_network[activation.node] {
                    ReteNode::Beta { tokens } => tokens.insert(0, new_token),
                    _ => unreachable!("tokens can only be stored in beta nodes"),
                }
            }

            for (production, token) in activations {
                match &mut self.beta_network[production] {
                    ReteNode::P { activations, .. } => activations.push(*token),
                    _ => unreachable!(),
                }
            }
        }
    }

    fn join_test(&self, test: &JoinNodeTest, token: TokenId, wme: &Wme) -> bool {
        let token = (0..test.beta_condition_offset).fold(token, |token, _| {
            self.tokens
                .neighbors_directed(token, Direction::Incoming)
                .next()
                .unwrap()
        });
        let other_wme = self.tokens[token].wme;

        wme.0[test.alpha_field] == other_wme.0[test.beta_field]
    }

    #[cfg(test)]
    fn network_graph(&self) -> petgraph::Graph<String, &'static str> {
        use petgraph::visit::IntoNodeReferences;

        let mut graph = petgraph::Graph::new();
        let mut alpha_indices = HashMap::new();
        let mut indices = HashMap::new();

        for (test, id) in &self.alpha_tests {
            let node = &self.alpha_network[id];
            let value = format!(
                "{:?}\ntest: {:?}\nwmes: {}",
                node.id,
                test.0,
                node.wmes.len()
            );
            let index = graph.add_node(value);
            alpha_indices.insert(id, index);
        }

        for (id, node) in self.beta_network.node_references() {
            let value = format!(
                "id: {:?}\nchildren: {:?}\nkind: {}",
                id,
                self.beta_network.neighbors(id).collect::<Vec<ReteNodeId>>(),
                match node {
                    ReteNode::Beta { ref tokens, .. } => format!("beta - {} tokens", tokens.len()),
                    ReteNode::Join { ref tests, .. } => format!("join - {} tests", tests.len()),
                    ReteNode::P { production } => format!("p: {:?}", production),
                }
            );
            let index = graph.add_node(value);
            indices.insert(id, index);
        }

        for node in self.alpha_network.values() {
            let alpha_index = alpha_indices[&node.id];
            for successor in &node.successors {
                let successor_index = indices[successor];
                graph.add_edge(alpha_index, successor_index, "a");
            }
        }

        for id in self.beta_network.node_indices() {
            let index = indices[&id];
            for child in self.beta_network.neighbors(id) {
                let child_index = indices[&child];
                graph.add_edge(index, child_index, "b");
            }
        }
        graph
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct AlphaTest([Option<SymbolID>; 3]);

impl AlphaTest {
    fn matches(&self, wme: &Wme) -> bool {
        self.0[0].map_or(true, |s| s == wme.0[0])
            && self.0[1].map_or(true, |s| s == wme.0[1])
            && self.0[2].map_or(true, |s| s == wme.0[2])
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct AlphaMemoryId(usize);

struct AlphaMemory {
    id: AlphaMemoryId,
    // Also called `items` in the Doorenbos thesis.
    wmes: Vec<Wme>,
    successors: Vec<ReteNodeId>,
}

//////////

type ReteNodeId = petgraph::graph::NodeIndex;
// #[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
// struct ReteNodeId(usize);

// unsafe impl petgraph::graph::IndexType for ReteNodeId {
//     fn new(x: usize) -> Self {
//         ReteNodeId(x)
//     }
//     fn index(&self) -> usize {
//         self.0
//     }
//     fn max() -> Self {
//         Self::new(<usize as petgraph::graph::IndexType>::max())
//     }
// }

#[derive(Debug)]
enum ReteNode {
    Beta {
        tokens: Vec<TokenId>,
    },
    Join {
        alpha_memory: AlphaMemoryId,
        tests: Vec<JoinNodeTest>,
    },
    P {
        production: ProductionID,
        activations: Vec<TokenId>,
    },
}

#[derive(Debug, PartialEq)]
struct JoinNodeTest {
    // These two should be able to fit in a single byte.
    alpha_field: usize,
    beta_field: usize,

    beta_condition_offset: usize,
}

type TokenId = petgraph::graph::NodeIndex;
// #[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
// struct TokenId(usize);

#[derive(Clone, Copy, Debug)]
struct Token {
    wme: Wme,
    /// The rete node containing this token. Used for quick removals.
    node: ReteNodeId,
}

//////////

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ProductionID(usize);

#[derive(Debug)]
pub struct Production {
    id: ProductionID,
    conditions: Vec<Condition>,
}

#[derive(Clone, Copy, Debug)]
struct Condition([ConditionTest; 3]);

#[derive(Clone, Copy, Debug, PartialEq)]
struct VariableID(usize);

#[derive(Clone, Copy, Debug)]
enum ConditionTest {
    Constant(SymbolID),
    Variable(VariableID),
}

impl Condition {
    /// Returns an iterator over only the variable tests, along with
    /// their indices.
    fn variables(&self) -> impl Iterator<Item = (usize, VariableID)> + '_ {
        self.0
            .iter()
            .enumerate()
            .filter_map(|(i, test)| match test {
                ConditionTest::Variable(id) => Some((i, *id)),
                ConditionTest::Constant(_) => None,
            })
    }
}

impl From<Condition> for AlphaTest {
    fn from(condition: Condition) -> AlphaTest {
        AlphaTest([
            condition.0[0].into(),
            condition.0[1].into(),
            condition.0[2].into(),
        ])
    }
}

impl From<ConditionTest> for Option<SymbolID> {
    fn from(test: ConditionTest) -> Option<SymbolID> {
        match test {
            ConditionTest::Constant(id) => Some(id),
            ConditionTest::Variable(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init() -> slog::Logger {
        // let _ = env_logger::builder().is_test(true).try_init();

        let decorator = slog_term::TermDecorator::new().build();
        let drain = slog_term::CompactFormat::new(decorator).build().fuse();
        // let drain = slog_term::FullFormat::new(decorator).build().fuse();
        let drain = std::sync::Mutex::new(drain).fuse();
        let log = slog::Logger::root(drain, o! {});

        log
    }

    #[test]
    fn empty_rete() {
        let rete = Rete::new(None);
        assert_eq!(rete.alpha_network.len(), 0);
        assert_eq!(rete.beta_network.node_count(), 1); // dummy top node.
    }

    #[allow(dead_code, unused_variables)]
    mod blocks {
        use super::*;

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

            assert_eq!(rete.alpha_tests.len(), 5);
            assert_eq!(rete.alpha_network.len(), 5);
            assert_eq!(rete.beta_network.node_count(), 13);
        }

        #[test]
        fn add_productions_and_wmes() {
            let log = init();

            let mut rete = Rete::new(log);
            for p in productions() {
                rete.add_production(p);
            }

            for wme in wmes() {
                rete.add_wme(wme);
            }

            let graph = rete.network_graph();
            let buffer = format!("{}", petgraph::dot::Dot::new(&graph));
            std::fs::write("graph.dot", buffer).ok();

            assert!(rete
                .take_events()
                .iter()
                .any(|event| *event == Event::Fired(ProductionID(1))));
        }

        #[test]
        fn add_wmes_then_productions() {
            let log = init();

            let mut rete = Rete::new(log);
            for wme in wmes() {
                rete.add_wme(wme);
            }
            for p in productions() {
                rete.add_production(p);
            }

            let graph = rete.network_graph();
            let buffer = format!("{}", petgraph::dot::Dot::new(&graph));
            std::fs::write("graph-wmes-then-productions.dot", buffer).ok();

            assert!(rete
                .take_events()
                .iter()
                .any(|event| *event == Event::Fired(ProductionID(1))));
        }
    }
}
