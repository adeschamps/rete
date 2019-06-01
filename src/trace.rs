use serde::{Deserialize, Serialize};
use slog::{Drain, Key, OwnedKVList, Record, SerdeValue, Serializer, KV};
use std::fmt;

/// A description of a modification that happened in the rete.
///
/// The rete will emit values of this type via its logging
/// infrastructure.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum Trace {
    Initialized,
    WmeAdded {
        id: usize,
    },
    WmeRemoved {
        id: usize,
    },
    ProductionAdded {
        id: usize,
    },
    ProductionRemoved {
        id: usize,
    },
    TokenAdded {
        node_id: usize,
        token_id: usize,
        parent_token_id: usize,
    },
    TokenRemoved {
        token_id: usize,
    },
    NodeAdded {
        id: usize,
        parent_id: usize,
        kind: (),
    },
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
