//! This crate implements the rete pattern matching algorithm.
//!
//! The rete is a data structure and algorithm for efficiently
//! detecting a large number of specific patterns in a graph
//! structure. The implementation here is based largely on Robert
//! Doorenbos' thesis, "Production Matching for Large Learning
//! Systems".

#[macro_use]
extern crate slog;

use petgraph::{
    stable_graph::{NodeIndex, StableGraph},
    Directed, Direction,
};
use slog::{Drain, Logger};
use std::{
    collections::{hash_map::Entry, HashMap},
    fmt,
    sync::atomic::{AtomicUsize, Ordering},
};

#[cfg(feature = "trace")]
pub mod trace;
#[cfg(feature = "trace")]
use trace::Trace;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

macro_rules! observe {
    ($log:ident, $event:expr) => {
        #[cfg(feature = "trace")]
        info!($log, "trace event"; $event);
    }
}

/// The type used to represent symbols. This may become a generic type parameter in the future.
pub type SymbolID = usize;

/// A working memory element.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Wme(pub [SymbolID; 3]);

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

/// A patten matcher using the rete algorithm.
pub struct Rete {
    log: Logger,
    alpha_tests: HashMap<AlphaTest, AlphaMemoryId>,
    alpha_network: HashMap<AlphaMemoryId, AlphaMemory>,
    beta_network: StableGraph<ReteNode, (), Directed>,
    dummy_node_id: ReteNodeId,
    tokens: StableGraph<Token, (), Directed>,

    productions: HashMap<ProductionID, ReteNodeId>,

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
    /// Construct a new rete. Optionally, a `slog::Logger` may be
    /// passed in. If it is not provided, then the rete will fall back
    /// to the standard log using the `slog_stdlog` crate.
    ///
    /// ```
    /// # use rete::Rete;
    /// // Construct a rete that logs to standard log.
    /// let rete = Rete::new(None);
    /// ```
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

        observe!(
            log,
            Trace::Initialized {
                dummy_node_id: dummy_node_id.index(),
                dummy_token_id: dummy_token_id.index(),
            }
        );

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

    /// Insert a WME into working memory.
    ///
    /// ```
    /// # use rete::{Rete, Wme};
    /// let mut rete = Rete::default();
    /// rete.add_wme(Wme([0, 1, 2]));
    /// ```
    pub fn add_wme(&mut self, wme: Wme) {
        let log = self.log.new(o!("wme" => format!("{:?}", wme)));
        trace!(log, "add wme");

        observe!(
            log,
            Trace::AddedWme {
                timetag: 0,
                id: wme.0[0],
                attribute: wme.0[1],
                value: wme.0[2],
            }
        );

        #[rustfmt::skip]
        let tests = [
            AlphaTest([None,           None,           None          ]),
            AlphaTest([None,           None,           Some(wme.0[2])]),
            AlphaTest([None,           Some(wme.0[1]), None          ]),
            AlphaTest([None,           Some(wme.0[1]), Some(wme.0[2])]),
            AlphaTest([Some(wme.0[0]), None,           None          ]),
            AlphaTest([Some(wme.0[0]), None,           Some(wme.0[2])]),
            AlphaTest([Some(wme.0[0]), Some(wme.0[1]), None          ]),
            AlphaTest([Some(wme.0[0]), Some(wme.0[1]), Some(wme.0[2])]),
        ];

        // Ensure we have an entry for this WME, even if no alpha
        // memories contain it.
        let _ = self.wme_alpha_memories.entry(wme).or_default();
        let _ = self.wme_tokens.entry(wme).or_default();

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

    /// Remove a WME from working memory.
    ///
    /// ```
    /// # use rete::{Rete, Wme};
    /// let mut rete = Rete::default();
    /// let wme = Wme([0, 1, 2]);
    /// rete.add_wme(wme);
    /// rete.remove_wme(wme);
    /// ```
    pub fn remove_wme(&mut self, wme: Wme) {
        let log = self.log.new(o!("wme" => format!("{:?}", wme)));
        trace!(log, "remove wme");

        let alpha_memories = self
            .wme_alpha_memories
            .remove(&wme)
            .expect("removing a WME that is not in any alpha memories.");
        for memory in alpha_memories {
            let alpha_memory = self.alpha_network.get_mut(&memory).unwrap();
            alpha_memory.wmes.retain(|w| *w != wme);
        }

        let mut tokens_to_remove = self
            .wme_tokens
            .remove(&wme)
            .expect("removing a WME that is not in any tokens.");
        while let Some(token_id) = tokens_to_remove.pop() {
            tokens_to_remove.extend(self.tokens.neighbors(token_id));
            let token = self.tokens.remove_node(token_id).unwrap();
            match self.beta_network[token.node] {
                ReteNode::Beta { ref mut tokens } => tokens.retain(|t| *t != token_id),
                _ => unreachable!("attempt to remove a WME from a non-beta node"),
            }
        }
    }

    /// Add a production to the rete. No reordering of conitions is performed.
    ///
    /// ```
    /// # use rete::*;
    /// # use rete::ConditionTest::*;
    /// let mut rete = Rete::default();
    ///
    /// // Add a production that includes a variable match:
    /// //     (0 ^1 <a>)
    /// //     (<a> ^2 3)
    /// let production = Production {
    ///     id: ProductionID(0),
    ///     conditions: vec![
    ///         Condition([Constant(0), Constant(1), Variable(VariableID(0))]),
    ///         Condition([Variable(VariableID(0)), Constant(2), Constant(3)]),
    ///     ],
    /// };
    /// rete.add_production(production);
    ///
    /// // Add an initial WME. The production should not match yet.
    /// rete.add_wme(Wme([0, 1, 10]));
    /// assert_eq!(rete.take_events(), vec![]);
    ///
    /// // Add a second WME that will cause the production to match.
    /// rete.add_wme(Wme([10, 2, 3]));
    /// assert_eq!(rete.take_events(), vec![Event::Fired(ProductionID(0))]);
    /// ```
    pub fn add_production(&mut self, production: Production) {
        let log = self.log.new(o!("production_id" => production.id.0));
        trace!(log, "add production"; "production" => ?production);

        if production.conditions.is_empty() {
            error!(log, "production has no conditions");
            return;
        }

        let mut current_node_id = self.dummy_node_id;

        for i in 0..production.conditions.len() {
            let condition = production.conditions[i];
            trace!(log, "add condition: {:?}", condition);

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

                observe!(
                    log,
                    Trace::AddedAlphaMemory {
                        id: alpha_memory_id.0,
                        test: trace::AlphaMemoryTest {
                            id: alpha_test.0[0],
                            attribute: alpha_test.0[1],
                            value: alpha_test.0[2],
                        }
                    }
                );
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

                    observe!(
                        log,
                        Trace::AddedNode {
                            id: id.index(),
                            parent_id: current_node_id.index(),
                            kind: trace::NodeKind::Join,
                            children: vec![],
                            alpha_node_id: Some(alpha_memory_id.0),
                        }
                    );

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

                        observe!(
                            log,
                            Trace::AddedNode {
                                id: id.index(),
                                parent_id: current_node_id.index(),
                                kind: trace::NodeKind::Beta,
                                children: vec![],
                                alpha_node_id: None,
                            }
                        );

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
        self.productions.insert(production.id, id);
        if current_node_id != self.dummy_node_id {
            self.activate_new_node(log.clone(), current_node_id, id);
        }
        trace!(log, "new p node"; "id" => ?id);

        observe!(
            log,
            Trace::AddedNode {
                id: id.index(),
                parent_id: current_node_id.index(),
                kind: trace::NodeKind::P,
                children: vec![],
                alpha_node_id: None,
            }
        );

        observe!(
            log,
            Trace::AddedProduction {
                id: production.id.0,
                p_node_id: id.index(),
            }
        );
    }

    pub fn remove_production(&mut self, id: ProductionID) {
        let log = self.log.new(o!("production" => id.0));
        trace!(log, "remove production");

        let mut current_node_id = match self.productions.remove(&id) {
            Some(id) => id,
            None => return,
        };

        while current_node_id != self.dummy_node_id
            && self.beta_network.neighbors(current_node_id).count() == 0
        {
            let log = log.new(o!("node" => current_node_id.index()));
            match self.beta_network[current_node_id] {
                ReteNode::Beta { ref tokens } => {
                    for token_id in tokens {
                        self.tokens.remove_node(*token_id);
                        trace!(log, "remove token"; "token" => token_id.index());
                        observe!(
                            log,
                            Trace::RemovedToken {
                                token_id: token_id.index()
                            }
                        );
                    }
                }
                ReteNode::Join { alpha_memory, .. } => {
                    match self.alpha_network.entry(alpha_memory) {
                        Entry::Occupied(mut alpha_node) => {
                            let index = alpha_node
                                .get()
                                .successors
                                .iter()
                                .position(|&id| id == current_node_id)
                                .expect("Alpha <-> join links are not consistent");
                            alpha_node.get_mut().successors.remove(index);
                            if alpha_node.get().successors.is_empty() {
                                alpha_node.remove();
                                trace!(log, "remove alpha memory"; "alpha memory" => alpha_memory.0);
                                observe!(log, Trace::RemovedAlphaMemory { id: alpha_memory.0 });
                            }
                        }
                        Entry::Vacant(_) => panic!("Join node's alpha memory does not exist"),
                    }
                }
                ReteNode::P { .. } => {}
            }

            let parent = self
                .beta_network
                .neighbors_directed(current_node_id, Direction::Incoming)
                .next()
                .expect("Node should have a parent");
            self.beta_network.remove_node(current_node_id);

            trace!(log, "remove node");
            observe!(
                log,
                Trace::RemovedNode {
                    id: current_node_id.index()
                }
            );

            current_node_id = parent;
        }

        self.productions.remove(&id);

        observe!(log, Trace::RemovedProduction { id: id.0 });
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
            let log = log.new(o!("activation" => format!("{:?}", activation)));
            let node = &self.beta_network[activation.node];
            trace!(log, "activating"; "remaining" => self.pending_activations.len());
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
                    observe!(
                        log,
                        Trace::AddedToken {
                            id: new_token_id.index(),
                            node_id: new_token.node.index(),
                            parent_id: token.index(),
                        }
                    );
                    new_tokens.push(new_token_id);
                    self.wme_tokens.get_mut(wme).unwrap().push(new_token_id);
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
                    observe!(
                        log,
                        Trace::MatchedProduction {
                            id: production.0,
                            token: token.index(),
                        }
                    );
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
            let value = format!("{:?}\ntest: {}\nwmes: {}", node.id, test, node.wmes.len());
            let index = graph.add_node(value);
            alpha_indices.insert(id, index);
        }

        for (id, node) in self.beta_network.node_references() {
            let value = format!(
                "id: {:?}\nchildren: {:?}\nkind: {}",
                id,
                self.beta_network
                    .neighbors(id)
                    .map(|id| id.index())
                    .collect::<Vec<_>>(),
                match node {
                    ReteNode::Beta { ref tokens, .. } => format!(
                        "beta - tokens: {:?}",
                        tokens.iter().map(|token| token.index()).collect::<Vec<_>>()
                    ),
                    ReteNode::Join { ref tests, .. } => format!("join - {} tests", tests.len()),
                    ReteNode::P {
                        production,
                        activations,
                    } => format!(
                        "p: {} - tokens: {:?}",
                        production.0,
                        activations
                            .iter()
                            .map(|token| token.index())
                            .collect::<Vec<_>>()
                    ),
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

    #[cfg(test)]
    fn token_graph(&self) -> StableGraph<String, &'static str> {
        self.tokens.map(
            |index, token| format!("token {}\n{:?}\n{:?}", index.index(), token.wme, token.node),
            |_, _| "",
        )
    }

    #[cfg(test)]
    fn write_graphs(&self) {
        let graph = self.network_graph();
        let buffer = format!("{}", petgraph::dot::Dot::new(&graph));
        std::fs::write("network.dot", buffer).ok();

        let graph = self.token_graph();
        let buffer = format!("{}", petgraph::dot::Dot::new(&graph));
        std::fs::write("tokens.dot", buffer).ok();
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct AlphaTest([Option<SymbolID>; 3]);

impl fmt::Display for AlphaTest {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let t = |test| match test {
            Some(symbol) => format!("{}", symbol),
            None => format!("*"),
        };
        write!(f, "({}, {}, {})", t(self.0[0]), t(self.0[1]), t(self.0[2]))
    }
}

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

#[derive(Clone, Copy, Debug)]
struct Token {
    wme: Wme,
    /// The rete node containing this token. Used for quick removals.
    node: ReteNodeId,
}

//////////

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ProductionID(pub usize);

#[derive(Debug)]
pub struct Production {
    pub id: ProductionID,
    pub conditions: Vec<Condition>,
}

#[derive(Clone, Copy, Debug)]
pub struct Condition(pub [ConditionTest; 3]);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VariableID(pub usize);

#[derive(Clone, Copy, Debug)]
pub enum ConditionTest {
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
        color_backtrace::install();

        #[cfg(not(feature = "trace"))]
        let drain = {
            let decorator = slog_term::TermDecorator::new().build();
            slog_term::CompactFormat::new(decorator).build().fuse()
        };

        #[cfg(feature = "trace")]
        let drain = trace::ObserverDrain::new(|msg| println!("{:?}", msg));

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

            rete.write_graphs();

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

            rete.write_graphs();

            assert!(rete
                .take_events()
                .iter()
                .any(|event| *event == Event::Fired(ProductionID(1))));
        }

        #[test]
        fn add_and_remove_wmes() {
            let log = init();

            let mut rete = Rete::new(log);
            for wme in wmes() {
                rete.add_wme(wme);
            }
            for wme in wmes() {
                rete.remove_wme(wme);
            }

            assert!(rete.wme_alpha_memories.is_empty());
            assert!(rete.wme_tokens.is_empty());
        }

        #[test]
        fn remove_productions() {
            let log = init();

            let mut rete = Rete::new(log);

            for p in productions() {
                rete.add_production(p);
            }
            for p in productions() {
                rete.remove_production(p.id);
            }

            assert!(rete.productions.is_empty());
            assert!(rete.alpha_network.is_empty());
            assert_eq!(rete.beta_network.node_count(), 1);
        }

        #[test]
        fn remove_productions_with_wmes() {
            let log = init();

            let mut rete = Rete::new(log);

            for p in productions() {
                rete.add_production(p);
            }
            for wme in wmes() {
                rete.add_wme(wme);
            }
            for p in productions() {
                rete.remove_production(p.id);
            }

            // The rete should just contain the dummy token.
            assert_eq!(rete.tokens.node_count(), 1);
        }

        #[test]
        #[ignore]
        fn duplicate_tokens() {
            unimplemented!()
        }
    }
}
