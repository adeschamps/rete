#![deny(missing_docs)]

//! Web assembly bindings to the rete. Requires `target_arch = "wasm32"`.

#[cfg(feature = "trace")]
use std::sync::{Arc, Mutex};
use wasm_bindgen::prelude::*;

/// A wrapper around the rete that can be exposed to wasm.
#[wasm_bindgen]
pub struct Rete {
    inner: crate::Rete,
    wmes: Vec<crate::Wme>,
    #[cfg(feature = "trace")]
    events: Arc<Mutex<Vec<crate::trace::Trace>>>,
    #[cfg(feature = "trace")]
    observers: Vec<Observer>,
}

#[wasm_bindgen]
impl Rete {
    /// Construct a new rete.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();

        #[cfg(feature = "trace")]
        let rete = {
            use slog::Drain;

            let events: Arc<Mutex<Vec<crate::trace::Trace>>> = Arc::new(Mutex::new(vec![]));

            let drain = crate::trace::ObserverDrain::new({
                let events = events.clone();
                move |msg| {
                    events.lock().unwrap().push(msg.clone());
                }
            });
            let log = slog::Logger::root(drain.fuse(), slog::o!());

            Self {
                inner: crate::Rete::new(log),
                wmes: vec![],
                events,
                observers: vec![],
            }
        };

        #[cfg(not(feature = "trace"))]
        let rete = Self {
            inner: crate::Rete::new(None),
            wmes: vec![],
        };

        rete
    }

    /// Register an object that will receive callbacks whenever events
    /// occur in the rete. Requires the `trace` feature.
    ///
    /// The `observer` should be an object with a `on_event` function
    /// which takes a string is its sole argument. This string can be
    /// parsed as JSON and interpreted according to the `Trace`
    /// type.
    #[cfg(feature = "trace")]
    pub fn observe(&mut self, observer: Observer) {
        self.observers.push(observer);
        self.dispatch();
    }

    /// Insert a WME into the rete.
    pub fn add_wme(&mut self, id: usize, attribute: usize, value: usize) -> usize {
        let wme = crate::Wme([id, attribute, value]);
        self.inner.add_wme(wme);
        let wme_id = self.wmes.len();
        self.wmes.push(wme);

        self.dispatch();

        wme_id
    }

    /// Remove the WME with the given timetag.
    pub fn remove_wme(&mut self, wme_id: usize) {
        let wme = self.wmes[wme_id];
        self.inner.remove_wme(wme);

        self.dispatch();
    }

    /// Add a production to the rete.
    pub fn add_production(&mut self, production: Production) {
        let conditions = production
            .conditions
            .iter()
            .map(|condition| {
                let id = condition.id.into();
                let attribute = condition.attribute.into();
                let value = condition.value.into();
                crate::Condition([id, attribute, value])
            })
            .collect();

        let input = crate::Production {
            id: crate::ProductionID(production.id),
            conditions,
        };
        self.inner.add_production(input);

        self.dispatch();
    }

    /// Remove the production with the given id.
    pub fn remove_production(&mut self, production_id: usize) {
        let id = crate::ProductionID(production_id);
        self.inner.remove_production(id);

        self.dispatch();
    }

    /// Notify all observers of recent events. They are passed a JSON
    /// string, which should be parsed on the JavaScript side
    /// according to the definition of the `Trace` enum.
    ///
    /// If the `trace` feature is not enabled, then this is a no-op.
    fn dispatch(&mut self) {
        #[cfg(feature = "trace")]
        for msg in self.events.lock().unwrap().drain(..) {
            let buffer = serde_json::to_string(&msg).unwrap();
            for observer in &self.observers {
                observer.on_event(&buffer);
            }
        }
    }
}

/// A complete production.
#[wasm_bindgen]
pub struct Production {
    id: usize,
    conditions: Vec<Condition>,
}

#[wasm_bindgen]
impl Production {
    /// Construct a new production with an ID but no conditions. Add
    /// conditions using the `add_condition` method.
    #[wasm_bindgen(constructor)]
    pub fn new(id: usize) -> Self {
        Self {
            id,
            conditions: vec![],
        }
    }

    /// Add a condition to the production.
    pub fn add_condition(&mut self, condition: Condition) {
        self.conditions.push(condition);
    }
}

/// A condition test for a single WME.
#[wasm_bindgen]
pub struct Condition {
    id: Test,
    attribute: Test,
    value: Test,
}

#[wasm_bindgen]
impl Condition {
    /// Construct a new condition that tests an id, attribute, and
    /// value.
    #[wasm_bindgen(constructor)]
    pub fn new(id: Test, attribute: Test, value: Test) -> Self {
        Self {
            id,
            attribute,
            value,
        }
    }
}

/// A test for a single symbol.
///
/// Tests can be either a value test or a variable binding. For value
/// tests, the ID is relative to all the other symbols in the Rete.
/// For variable tests, the ID is relative to other variables within
/// the same production; that is, multiple productions can reuse the
/// same variable IDs.
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct Test {
    symbol: usize,
    is_variable: bool,
}

#[wasm_bindgen]
impl Test {
    /// Construct a new value test. The first parameter is either a
    /// symbol ID (scoped to the Rete) or a variable ID (scoped to the
    /// production). The second parameter indicates which type of test
    /// this is.
    #[wasm_bindgen(constructor)]
    #[deprecated = "Use the `Test::symbol` and `Test::variable` constructors instead."]
    pub fn new(symbol: usize, is_variable: bool) -> Self {
        Self {
            symbol,
            is_variable,
        }
    }

    /// Construct a new test for a constant symbol. Symbol IDs within
    /// the same Rete refer to the same values.
    pub fn symbol(symbol: usize) -> Self {
        Self {
            symbol,
            is_variable: false,
        }
    }

    /// Construct a new test for a variable. Variable IDs are scoped
    /// to a production; IDs within the same production must match,
    /// but different productions can reuse the same variable IDs.
    pub fn variable(id: usize) -> Self {
        Self {
            symbol: id,
            is_variable: true,
        }
    }
}

impl Into<crate::ConditionTest> for Test {
    fn into(self) -> crate::ConditionTest {
        if self.is_variable {
            let id = crate::VariableID(self.symbol);
            crate::ConditionTest::Variable(id)
        } else {
            crate::ConditionTest::Constant(self.symbol)
        }
    }
}

#[cfg(feature = "trace")]
#[wasm_bindgen]
extern "C" {
    /// Binding to an object that will receive rete events. Requires
    /// the `trace` feature.
    #[wasm_bindgen(js_name = "Observer")]
    pub type Observer;

    /// Called whenever an event occurs in the rete. The argument
    /// passed to this function is a JSON string serialized from the
    /// `trace::Trace` enum.
    #[wasm_bindgen(method)]
    pub fn on_event(this: &Observer, event_json: &str);
}
